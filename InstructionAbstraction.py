import sys
import os
import random
import re
from collections import namedtuple
import setting as config
import copy

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
    def __init__(self, opcode, input, output, funcname="", address=-1):
        self.opcode = opcode
        self.input = input
        self.output = output
        self.address = address
        self.funcname = funcname

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
                if " -> " in item:
                    inst_key.append(item.split(" -> ")[1])
                if "Opcode" in item:
                    opcode = item.split(": ")[1]
                if "name" in item:
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
                                    res = re.findall("GepType\+\d+", item)
                                    type = str(type)+"+"+res[0].split("+")[1]
                            value = ""
                            operand = op.split(" ")[1]
                            operand = operand.lstrip(" ")
                            #if operand.startswith("%"):
                            #    operand = "bo"+operand
                            if opcode == "alloca":
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
                                    res = re.findall("GepType\+\d+", item)
                                    type = str(type)+"+"+res[0].split("+")[1]
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
        inst_key = []
        inst_value = []
        trace = []
        self.parseIRInst(self.static_trace, inst_key, inst_value)
        inst_map = dict(zip(inst_key,inst_value))
        remap = []
        memoryBoundary = []
        Data_item = namedtuple("data_item", "operand type value")
        for idx, item in enumerate(self.dynamic_trace):
            if "ID:" in item:
                item_new = item.rstrip("\n")
                res = re.findall("[-+]?[0-9]*\.?[0-9]+", item_new)
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
                trace.append(value)
                remap.append(idx)
            else:
                memoryBoundary.append(idx)
        memory = self.processMemory(memoryBoundary)
        ret = []
        ret.append(trace)
        ret.append(remap)
        ret.append(memory)
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
        for key in memory.keys():
            maps = memory[key]
            memRange[key] = []
            for i in maps:
                if "profiling.exe" in i or "[heap]" in i or "[stack]" in i or "[stack:"+config.tid+"]" in i:
                    res = re.findall("[0-9a-fA-F]+-[0-9a-fA-F]+", i)
                    min = int(res[0].split("-")[0], 16)
                    max = int(res[0].split("-")[1], 16)
                    memRange[key].append(min)
                    memRange[key].append(max)
        return memRange





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
                for arg in args:
                    arg = arg.lstrip("(").rstrip(")")
                items = line.split(" ")
                if "void" in line:
                    if funcname not in func.keys():
                        func[funcname] = []
                        func[funcname].append("void")
                else:
                    if funcname not in func.keys():
                        func[funcname] = []
                        func[funcname].append(items[1])
                for arg in args:
                    if "%" in arg:
                        func[funcname].append(arg)
        return func



