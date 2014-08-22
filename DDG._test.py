import os
import sys
import re
import random
from networkx import graphviz_layout

import InstructionAbstraction
import networkx as nx
import matplotlib.pyplot as plt
import setting as config
import PVF as pvf

counter = 0
phiNodeCheck = {}

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


class DDG:
    @classmethod
    def __init__(self, trace):
        self.dynamic_trace = trace

    # ##
    # A node can be 1. source 2. dest 3. opcode
    # ##
    @classmethod
    def ddg_construct(self, trace, memcpyRec, bitwiseRec):
        global counter
        global phiNodeCheck
        global blacklist
        G = nx.DiGraph()
        multiInstance = {}
        rename_mapping = {}
        fm = InstructionAbstraction.FunctionMapping(config.IRpath)
        funcMap = fm.extractFuncDef()
        # format of funcmap: {funcname:[type/void, arg1, arg2 ...], funcname:[...]}
        flag_phi = 0
        totalbits = 0
        for idx, ddg_inst in enumerate(trace):
            # the following two lists hold the newly added nodes for the current instruction
            # the nodes can be either a operand or a memory address
            source_node = []
            dest_node = []
            instbits = 0
            for source in ddg_inst.input:
                op = source.operand
                if op in rename_mapping.keys():
                    op = rename_mapping[op]
                if op not in multiInstance.keys():
                    multiInstance[op] = op
                else:
                    op = multiInstance[op]
                itype = source.type
                if ddg_inst.opcode not in config.memoryInst:
                    res = re.findall('\d+', itype)
                    instbits += int(res[0])
                if ddg_inst.address != -1:
                    if ddg_inst.address not in multiInstance.keys():
                        multiInstance[ddg_inst.address] = ddg_inst.address
                    else:
                        ddg_inst.address = multiInstance[ddg_inst.address]
                    if ddg_inst.opcode == "load":
                        if ddg_inst.address not in G.nodes():
                            G.add_node(ddg_inst.address, len=itype, size=1, operand0=op)
                            G.add_node(op, len=itype,size=1, operand0=op, dest=ddg_inst.output[0].operand)
                            G.add_edge(ddg_inst.address, op, opcode='virtual')
                            G.add_edge(op, ddg_inst.address, opcode='virtual')
                            source_node.append(ddg_inst.address)
                        else:
                            size = int(G.node[ddg_inst.address]['size']) + 1
                            G.node[ddg_inst.address]['size'] = size
                            G.node[ddg_inst.address]['operand' + str(size - 1)] = op
                            source_node.append(ddg_inst.address)
                            # create fake edges between the address and the register
                            G.add_node(op, len=itype,size=1, operand0=op, dest=ddg_inst.output[0].operand)
                            G.add_edge(ddg_inst.address, op, opcode='virtual')
                            G.add_edge(op, ddg_inst.address, opcode='virtual')
                            # addr_op_map[ddg_inst.address] = op
                    elif ddg_inst.opcode == "call":
                        if ddg_inst.funcname in funcMap.keys():
                            op_rep = funcMap[ddg_inst.funcname][ddg_inst.index(op) + 1]
                            rename_mapping[op_rep] = op
                    else:
                        #if op not in G.nodes():
                        flag = 0
                        for node in G.nodes():
                            for i in range(int(G.node[node]['size'])):
                                if G.node[node]['operand' + str(i)] == op:
                                    source_node.append(node)
                                    flag = 1
                                    break
                        if flag == 0:
                            if isint(op) or isfloat(op):
                                counter += 1
                                op_new = "constant" + str(counter)
                                G.add_node(op_new, len=itype, size=1, operand0=op_new)
                                source_node.append(op_new)
                            else:
                                G.add_node(op, len=itype, size=1, operand0=op)
                                source_node.append(op)
                        #else:
                        #    source_node.append(op)
                else:
                    if ddg_inst.opcode == "call":
                        if ddg_inst.funcname in funcMap.keys():
                            op_rep = funcMap[ddg_inst.funcname][ddg_inst.index(op) + 1]
                            rename_mapping[op_rep] = op
                        if "memcpy" in ddg_inst.funcname:
                            # if op not in G.nodes():
                            #    flag = 0
                            #    for node in G.nodes():
                            #        for i in range(int(G.node[node]['size'])):
                            #            if G.node[node]['operand'+str(i)] == op:
                            #                source_node.append(node)
                            #                flag = 1
                            #                break
                            #    if flag == 0:
                            #        G.add_node(op, len=itype, size=1, operand0=op)
                            #        source_node.append(op)
                            #else:
                            #    source_node.append(op)
                            source_address = memcpyRec[idx][1]
                            length = memcpyRec[idx][2]
                            for i in range(int(length) / 4):
                                if str(int(source_address) + i * 4) in G.nodes():
                                    size = int(G.node[str(int(source_address) + i * 4)]['size']) + 1
                                    G.node[str(int(source_address) + i * 4)]['size'] = size
                                    G.node[str(int(source_address) + i * 4)]['operand' + str(size - 1)] = op
                                    source_node.append(str(int(source_address) + i * 4))
                                else:
                                    G.add_node(str(int(source_address) + i * 4), len=itype, size=1, operand0="")
                                    source_node.append(str(int(source_address) + i * 4))
                    else:
                        #if op not in G.nodes():
                            if ddg_inst.opcode == "phi":
                                for i in xrange(idx, 0, -1):
                                    if len(trace[i].output) > 0:
                                        if trace[i].output[0].operand == op:
                                            phiNodeCheck[i] = op
                                            flag_phi = 1
                            else:
                                flag = 0
                                for node in G.nodes():
                                    for i in range(int(G.node[node]['size'])):
                                        if G.node[node]['operand' + str(i)] == op:
                                            source_node.append(node)
                                            flag = 1
                                            break
                                if flag == 0:
                                    if isint(op) or isfloat(op):
                                        counter += 1
                                        op_new = "constant" + str(counter)
                                        G.add_node(op_new, len=itype, size=1, operand0=op_new)
                                        source_node.append(op_new)
                                    else:
                                        if ddg_inst.opcode == "and" or ddg_inst.opcode == "or" \
                                                or ddg_inst.opcode == "shl" or \
                                                        ddg_inst.opcode == "lshr" or ddg_inst.opcode == "ashr":
                                            G.add_node(op, len=itype, size=1,
                                                       operand0=op, bits=bitwiseRec[ddg_inst.input.index(source)])
                                            source_node.append(op)
                                        else:
                                            G.add_node(op, len=itype, size=1, operand0=op)
                                            source_node.append(op)
                        #else:
                            #if ddg_inst.opcode == "phi":
                            #    for i in xrange(idx, 0, -1):
                            #        if len(trace[i].output) > 0:
                            #            if trace[i].output[0].operand == op:
                            #                phiNodeCheck[i] = op
                            #                flag_phi = 1
                           # else:
                            #    source_node.append(op)
            if flag_phi == 1:
                index = max(phiNodeCheck.keys())
                source_node.append(phiNodeCheck[index])
                phiNodeCheck = {}
                flag_phi = 0

            for dest in ddg_inst.output:
                op = dest.operand
                if op in rename_mapping.keys():
                    op = rename_mapping[op]
                if op not in multiInstance.keys():
                    multiInstance[op] = op
                else:
                    op = multiInstance[op]
                itype = dest.type
                if ddg_inst.opcode not in config.memoryInst:
                    res = re.findall('\d+', itype)
                    instbits += int(res[0])
                if ddg_inst.address != -1:
                    if ddg_inst.address not in multiInstance.keys():
                        multiInstance[ddg_inst.address] = ddg_inst.address
                    else:
                        ddg_inst.address = multiInstance[ddg_inst.address]
                    if ddg_inst.opcode == "store" or ddg_inst.opcode == "alloca":
                        if ddg_inst.address not in G.nodes():
                            G.add_node(ddg_inst.address, len=itype, size=1, operand0=op)
                            dest_node.append(ddg_inst.address)
                            G.add_node(op, len=itype,size=1, operand0=op)
                            G.add_edge(ddg_inst.address, op, opcode='virtual')
                            G.add_edge(op, ddg_inst.address, opcode='virtual')
                        else:
                            counter += 1
                            ddg_inst.address = ddg_inst.address.split("+")[0]
                            ddg_inst.address1 = ddg_inst.address+"+"+str(counter)
                            multiInstance[ddg_inst.address] = ddg_inst.address1
                            G.add_node(ddg_inst.address + "+" + str(counter), len=itype, size=1, operand0=op)
                            dest_node.append(ddg_inst.address + "+" + str(counter))
                            G.add_node(op, len=itype,size=1, operand0=op)
                            G.add_edge(ddg_inst.address+ "+" + str(counter), op, opcode='virtual')
                            G.add_edge(op, ddg_inst.address+ "+" + str(counter), opcode='virtual')
                            # addr_op_map[ddg_inst.address] = op
                    elif ddg_inst.opcode == "call":
                        if ddg_inst.funcname in funcMap.keys():
                            op_rep = funcMap[ddg_inst.funcname][ddg_inst.index(op) + 1]
                            rename_mapping[op_rep] = op
                    else:
                        if op not in G.nodes():
                            flag = 0
                            for node in G.nodes():
                                for i in range(int(G.node[node]['size'])):
                                    if G.node[node]['operand' + str(i)] == op:
                                        dest_node.append(node)
                                        flag = 1
                                        break
                            if flag == 0:
                                G.add_node(op, len=itype, size=1, operand0=op)
                                dest_node.append(op)
                        else:
                            counter += 1
                            op = op.split("+")[0]
                            op1 = op+"+"+str(counter)
                            multiInstance[op] = op1
                            G.add_node(op1, len=itype, size=1, operand0=op1)
                            dest_node.append(op + "+" + str(counter))
                else:
                    if ddg_inst.opcode == "call":
                        if ddg_inst.funcname in funcMap.keys():
                            op_rep = funcMap[ddg_inst.funcname][ddg_inst.index(op) + 1]
                            rename_mapping[op_rep] = op
                        if "memcpy" in ddg_inst.funcname:
                            # if op not in G.nodes():
                            #    flag = 0
                            #    for node in G.nodes():
                            #        for i in range(int(G.node[node]['size'])):
                            #            if G.node[node]['operand'+str(i)] == op:
                            #                dest_node.append(node)
                            #                flag = 1
                            #                break
                            #    if flag == 0:
                            #        G.add_node(op, len=itype, size=1, operand0=op)
                            #        dest_node.append(op)
                            #else:
                            #    dest_node.append(op)
                            dest_address = memcpyRec[idx][0]
                            length = memcpyRec[idx][2]
                            for i in range(int(length) / 4):
                                if str(int(dest_address) + i * 4) in G.nodes():
                                    size = int(G.node[str(int(dest_address) + i * 4)]['size']) + 1
                                    G.node[str(int(dest_address) + i * 4)]['size'] = size
                                    G.node[str(int(dest_address) + i * 4)]['operand' + str(size - 1)] = op
                                    dest_node.append(str(int(dest_address) + i * 4))
                                else:
                                    G.add_node(str(int(dest_address) + i * 4), len=itype, size=1, operand0="")
                                    dest_node.append(str(int(dest_address) + i * 4))
                    else:
                        #if op not in G.nodes():
                            flag = 0
                            for node in G.nodes():
                                for i in range(int(G.node[node]['size'])):
                                    if G.node[node]['operand' + str(i)] == op:
                                        dest_node.append(node)
                                        flag = 1
                                        break
                            if flag == 0:
                                if ddg_inst.opcode == "and" or ddg_inst.opcode == "or" or ddg_inst.opcode == "shl" or ddg_inst.opcode == "lshr" or ddg_inst.opcode == "ashr":
                                    G.add_node(op, len=itype, size=1, operand0=op, bits=bitwiseRec[2])
                                    dest_node.append(op)
                                else:
                                    G.add_node(op, len=itype, size=1, operand0=op)
                                    dest_node.append(op)
                        #else:
                           # counter += 1
                           # G.add_node(op + "+" + str(counter), len=itype, size=1, operand0=op)
                           # dest_node.append(op + "+" + str(counter))
            for s_node in source_node:
                for d_node in dest_node:
                    if "memcpy" in ddg_inst.funcname:
                        if source_node.index(s_node) == dest_node.index(d_node):
                            G.add_edge(s_node, d_node, opcode=ddg_inst.opcode)
                    else:
                        G.add_edge(s_node, d_node, opcode=ddg_inst.opcode)
            totalbits += instbits
        print totalbits
        return G


a = InstructionAbstraction.AbstractInst(config.indexFilePath, config.tracePath)
trace = a.export_trace()
ddg = DDG(trace)
G = ddg.ddg_construct(ddg.dynamic_trace, a.memcpyRec, a.bitwiseRec)
nx.draw_random(G)
#nx.write_dot(G, "./test.dot")
#pvf_res = pvf.PVF(G, trace)
#subG = pvf_res.computePVF(config.outputDataSet)
#nx.nx.write_dot(subG, "./subgraph.dot")
plt.show()