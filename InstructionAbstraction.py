import sys
import os
import random
import re
from collections import namedtuple
import setting as config
import copy
import math

gmem = {}

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

class DDGInst:
    def __init__(self, opcode, input, output, funcname="", address=-1, cycle = 0):
        self.opcode = opcode
        self.input = input
        self.output = output
        self.address = address
        self.funcname = funcname
        self.cycle = cycle

class AbstractInst:

    def __init__(self, mapping, trace):
        self.mapping = mapping
        self.trace = trace
        self.static_trace = []
        self.dynamic_trace = []
        with open(self.mapping, "r") as f_mapping:
            self.static_trace = f_mapping.readlines()
        with open(self.trace, "r") as f_trace:
            self.dynamic_trace = f_trace.readlines()
        self.memcpyRec = {}
        self.bitwiseRec = {}

    def parseIRInst(self,static_trace,inst_key,inst_value):
        flag = 0
        strace = []
        block = []
        Data_item = namedtuple("data_item", "operand type value")
        for item in static_trace:
            if item.strip() == "start":
                flag = 1
            elif item.strip() == "end":
                flag = 0
                strace.append(block)
                block = []
            elif flag == 1:
                block.append(item)
        for block in strace:
            opcode = ""
            input = []
            output = []
            funcname = ""
            outername = ""
            for item in block:
                item = item.rstrip("\n")
                if "_ZL12center_table" in item:
                    print "hhhh"
                if " -> " in item:
                    inst_key.append(item.split(" -> ")[1])
                if "Opcode" in item:
                    opcode = item.split(": ")[1]
                if "name:" in item and "Func" not in item:
                    outername = item.split(" ")[1]
                if "Source" in item:
                    if item != "Source: null":
                        if " = " not in item:
                            op = item.split(": ")[1]
                            #type = op.split(" ")[0]
                            res = re.findall("Type\+\d+", item)
                            type = int(res[0].split("+")[1])
                            if opcode == "getelementptr":
                                if "GepType+" in item:
                                    res = re.findall("GepType\+\d+\+?.*", item)
                                    new_res = res[0].split("+")
                                    if len(new_res) > 2:
                                        type = str(type)+"+"+new_res[1]+"+"+new_res[2]
                                    else:
                                        type = str(type)+"+"+new_res[1]
                            value = ""
                            operand = op.split(" ")[1]
                            operand = operand.lstrip(" ")
                            #if operand.startswith("%"):
                            #    operand = "bo"+operand
                            if opcode == "alloca":
                                type = config.OSbits
                                d = Data_item(outername+"_root"+"_"+str(random.randint(0,1024)),type, value)
                            else:
                                temp = ""
                                if isint(operand) or isfloat(operand):
                                    temp = operand
                                else:
                                    temp = outername+"_"+operand
                                d = Data_item(temp, type, value)
                        else:
                            op = item.split(": ")[1]
                            operand = op.split(" = ")[0].lstrip(" ")
                            #if operand.startswith("%"):
                            #    operand = "bo"+operand
                            #res = re.findall("i\d+\*?\*?", op.split(" = ")[1])
                            res = re.findall("Type\+\d+", item)
                            type = int(res[0].split("+")[1])
                            if opcode == "getelementptr":
                                if "GepType+" in item:
                                    res = re.findall("GepType\+\d+\+?.*", item)
                                    new_res = res[0].split("+")
                                    if len(new_res) > 2:
                                        type = str(type)+"+"+new_res[1]+"+"+new_res[2]
                                    else:
                                        type = str(type)+"+"+new_res[1]
                            temp = ""
                            value = ""
                            if isint(operand) or isfloat(operand):
                                temp = operand
                            else:
                                temp = outername+"_"+operand
                            d = Data_item(temp, type, value)
                            #if len(res) > 0:
                            #    d = Data_item(operand, type)
                            #else:
                            #    d = Data_item(operand, "0")
                        if "memcpy" in funcname:
                            if len(input) < 1:
                                input.append(d)
                        else:
                            input.append(d)
                if "Dest" in item:
                    if "unreachable" not in item and "void" not in item:
                        op = item.split(": ")[1]
                        operand = op.split(" = ")[0]
                        operand = operand.lstrip(" ")
                        #if operand.startswith("%"):
                        #        operand = "bo"+operand
                        #res = re.findall("i\d+\*?\*?", op.split(" = ")[1])
                        res = re.findall("Type\+\d+", item)
                        type = int(res[0].split("+")[1])
                        value = ""
                        #if " to " in item and len(res) > 1:
                        temp = ""
                        if isint(operand) or isfloat(operand):
                            temp = operand
                        else:
                            temp = outername+"_"+operand
                        d = Data_item(temp, type, value)
                        #else:
                        #    if len(res) > 0:
                        #        d = Data_item(operand, res[0])
                        #    else:
                        #        d = Data_item(operand, "0")
                        output.append(d)
                if "Funcname" in item:
                    funcname = item.split(": ")[1]

            ddg_inst = DDGInst(opcode, input, output, funcname)
            inst_value.append(ddg_inst)

    def export_trace(self):
        # ID: 312 OPCode: 26 Value: 139679839926848
        global gmem
        inst_key = []
        inst_value = []
        trace = []
        self.parseIRInst(self.static_trace, inst_key, inst_value)
        inst_map = dict(zip(inst_key,inst_value))
        remap = []
        memoryBoundary = []
        memory = {}
        Data_item = namedtuple("data_item", "operand type value")
        _count = 0
        cycle_index_lookup = {}
        index_cycle_lookup = {}
        for item in self.dynamic_trace:
            if _count == 2:
                break
            if "profiling.exe" in item:
                    res = re.findall("[0-9a-fA-F]+-[0-9a-fA-F]+", item)
                    min = int(res[0].split("-")[0], 16)
                    max = int(res[0].split("-")[1], 16)
                    gmem[_count] = []
                    gmem[_count].append(min)
                    gmem[_count].append(max)
                    _count += 1
        for idx, item in enumerate(self.dynamic_trace):
            if "ID:" in item:
                item_new = item.rstrip("\n")
                res = re.findall("[-+]?[0-9]*\.?[0-9]+", item_new)
                cycle = res[len(res)-1]
                res.pop()
                _index = int(res[0])
                if res[1] != "2" and res[1] != "48" and res[1] != "26":
                    cycle_index_lookup[int(cycle)] = _index
                    if _index not in index_cycle_lookup:
                        index_cycle_lookup[_index] = []
                        index_cycle_lookup[_index].append(cycle)
                    else:
                        index_cycle_lookup[_index].append(cycle)

                value = copy.deepcopy(inst_map[res[0]])
                assert isinstance(value, DDGInst)
                if len(value.output)>0:
                    output_operand = value.output[0].operand
                    output_type = value.output[0].type
                output_value = ""
                if res[1] == "27" or res[1] == "28" or res[1] == "26":
                #if len(res) > 2:
                  value.address = res[2]
                  if res[1] == "27" and len(res) == 4:
                      output_value = res[3]

                  if res[1] == "27" or res[1] == "28":
                      self.processLDST(memoryBoundary,memory,idx)
                      memoryBoundary = []
                else:
                    if len(res) >= 3 :
                        output_value = res[2]
                if len(res) == 5:
                    self.memcpyRec[idx] = []
                    self.memcpyRec[idx].append(res[2])
                    self.memcpyRec[idx].append(res[3])
                    self.memcpyRec[idx].append(res[4])
                if len(res) == 6:
                    self.bitwiseRec[idx] = []
                    self.bitwiseRec[idx].append(res[2])
                    self.bitwiseRec[idx].append(res[3])
                    self.bitwiseRec[idx].append(res[4])
                if len(value.output)>0:
                    value.output[0] = Data_item(output_operand, output_type, output_value)
                value.cycle = int(cycle)
                trace.append(value)
                remap.append(idx)
            else:
                memoryBoundary.append(idx)
        #memory = self.processMemory(memoryBoundary)
        ret = []
        ret.append(trace)
        ret.append(remap)
        ret.append(memory)
        ret.append(cycle_index_lookup)
        ret.append(index_cycle_lookup)
        return ret

    def processMemory(self, memoryBoundary):
        counter = -1
        memory = {}
        temp = []
        for item in memoryBoundary:
            counter += 1
            if counter != item:
                counter = item
                memory[item] = []
                memory[item].extend(temp)
                temp = []
            temp.append(self.dynamic_trace[item])
        memRange = {}
        for key in memory:
            maps = memory[key]
            memRange[key] = []
            for i in maps:
                if "profiling.exe" in i or "[heap]" in i or "[stack]" in i or "[stack:"+config.tid+"]" in i:
                    res = re.findall("[0-9a-fA-F]+-[0-9a-fA-F]+", i)
                    min = int(res[0].split("-")[0], 16)
                    max = int(res[0].split("-")[1], 16)
                    memRange[key].append(min)
                    memRange[key].append(max)
        print memRange
        return memRange

    def processLDST(self, memoryBoundary, g_memory, index_of_ldst):
        global gmem
        memory = []
        for item in memoryBoundary:
            memory.append(self.dynamic_trace[item])
        g_memory[index_of_ldst] = {}
        for iter, key in enumerate(gmem.keys()):
            g_memory[index_of_ldst][iter] = gmem[key]
            g_memory[index_of_ldst][iter] = gmem[key]
        for i in memory:
            if "[stack]" in i:
                res = re.findall("[0-9a-fA-F]+-[0-9a-fA-F]+", i)
                min = int(res[0].split("-")[0], 16)
                max = int(res[0].split("-")[1], 16)
                g_memory[index_of_ldst]["stack"] = []
                g_memory[index_of_ldst]["stack"].append(min)
                g_memory[index_of_ldst]["stack"].append(max)
            if "[heap]" in i:
                res = re.findall("[0-9a-fA-F]+-[0-9a-fA-F]+", i)
                min = int(res[0].split("-")[0], 16)
                max = int(res[0].split("-")[1], 16)
                g_memory[index_of_ldst]["heap"] = []
                g_memory[index_of_ldst]["heap"].append(min)
                g_memory[index_of_ldst]["heap"].append(max)
            if "esp" in i:
                esp = i.rstrip("\n").split(" ")[1]
                g_memory[index_of_ldst]["esp"] = []
                g_memory[index_of_ldst]["esp"].append(int(esp))






class FunctionMapping:

    def __init__(self, ir_file):
        with open(ir_file, "r") as irf:
            self.ir = irf.readlines()

    def extractFuncDef(self):
        func = {}
        for line in self.ir:
            if "define" in line:
                line = line.rstrip("\n")
                res = re.findall('@.*\(.*\)', line)
                funcname = re.findall('.*\(', res[0].lstrip("@"))
                funcname = funcname[0].rstrip(")")
                paras = re.findall('\(.*\)', res[0])
                args = paras[0].split(" ")
                items = line.split(" ")
                if "void" in line:
                    if funcname not in func:
                        func[funcname] = []
                        func[funcname].append("void")
                else:
                    if funcname not in func:
                        func[funcname] = []
                        func[funcname].append(items[1])
                for arg in args:
                    arg = arg.lstrip("(").rstrip(")")
                    if "%" in arg:
                        func[funcname].append(arg)
        return func

    def extractStruct(self):
        # %struct.anon = type { i32*, i32*, [48 x float]*, [48 x float]*, [48 x float]* }
        struct = {}
        for line in self.ir:
            if "struct" in line and "type" in line:
                line = line.rstrip("\n")
                str1 = line.split(" = ")
                name = str1[0]
                struct[name] = {}
                str2 = str1[1].lstrip("type { ").rstrip(" }")
                fields = str2.split(",")
                for idx, item in enumerate(fields):
                    type = 0
                    item1 = item
                    item1 = item1.lstrip(" ").rstrip(" ")
                    if "*" in item1:
                        type = config.OSbits
                    else:
                        if "i" in item1:
                            res = re.findall("[-+]?[0-9]*\.?[0-9]+",item1)
                            if len(res) == 0:
                                type = 0
                            elif len(res) == 2:
                                type = int(math.ceil(int(res[1]) / 4.0)) * 4
                            else:
                                type = int(math.ceil(int(res[0]) / 4.0)) * 4
                        if "float" in item1:
                            type = 32
                        if "double" in item1:
                            type = 64
                    struct[name][idx] = type
        return struct




