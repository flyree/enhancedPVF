import re
import sys
import os
import random
import networkx as nx
import copy
from itertools import izip_longest
import setting as config
import InstructionAbstraction
import sets
from multiprocessing import Manager, Process, current_process, Queue

aceBits = 0
crashBits = 0
stack = []
rangeList = {}
gcount = 0
finalBits = []


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class PVF:
    def __init__(self, G, trace, indexMap):
        self.G = G
        assert(len(trace) == 3)
        self.trace = trace[0]
        self.remap = trace[1]
        self.memory = trace[2]
        fm = InstructionAbstraction.FunctionMapping(config.IRpath)
        structMap = fm.extractStruct()
        self.structMap = structMap
        self.indexMap = indexMap


    def computePVF(self, targetList):

        #----------------
        # Get the predecessors of the target node
        #----------------
        global aceBits
        global crashBits
        predecessors = []
        for node in self.G:
            oprandlist = []
            opcode = ""
            for target in targetList:
                if target in node:
                    predecessors.append(node)
            for edge in self.G.in_edges(node):
                if "virtual" not in self.G.edge[edge[0]][edge[1]]['opcode']:
                    oprandlist.append(edge[0])
                    opcode = self.G.edge[edge[0]][edge[1]]['opcode']
            if opcode == "icmp" or opcode == "fcmp":
                    predecessors.extend(oprandlist)
                    predecessors.append(node)
        #for node in predecessors[:]:
        #    for edge in self.G.out_edges(node):
        #        if self.G.edge[edge[0]][edge[1]]['opcode'] != 'virtual':
        #            predecessors.remove(node)
        # i = 0
        # while i < len(predecessors):
        #     flag = 0
        #     newlist = self.G.predecessors(predecessors[i])
        #     offsets = {}
        #     for newnode in newlist:
        #        if 'dest' in self.G.node[newnode]:
        #            if self.G.node[newnode]['dest'] in predecessors:
        #                flag = 1
        #                offset = predecessors.index(self.G.node[newnode]['dest'])
        #                if offset not in offsets:
        #                    offsets[offset] = []:q
        #                    offsets[offset].append(newnode)
        #                else:
        #                    offsets[offset].append(newnode)
        #        else:
        #            if newnode not in predecessors:
        #                predecessors.append(newnode)
        #     if flag == 1:
        #         for node in offsets[sorted(offsets)[0]]:
        #             if node not in predecessors:
        #                 predecessors.append(node)
        #     print "nodes in predecessors: "+str(len(predecessors))+" "+str(i)+" "+str(len(newlist))
        #     assert(len(predecessors) <= self.G.number_of_nodes())
        #     i += 1
        ReG = self.G.reverse()
        sub_nodes = []
        print len(predecessors)
        print len(ReG.nodes())
        for target in predecessors:
          if target in ReG:
             T = nx.bfs_tree(ReG, target)
             tnodes = T.nodes()
             sub_nodes.extend(tnodes)
             ReG.remove_nodes_from(tnodes)
             #sub_nodes.extend(T.nodes())
        #processes = [Process(target=self.do_work, args=(ReG,target,sub_nodes)) for target in predecessors]
        #for p in processes:
        #    p.start()
        #for p in processes:
        #    p.join()
        subG = self.G.subgraph(sub_nodes)
        # source = []
        # for node in subG:
        #     if subG.in_degree(node) == 0:
        #         source.append(node)
        #     if subG.in_degree(node) == 1:
        #         for edge in subG.in_edges(node):
        #             if subG.edge[edge[0]][edge[1]]['opcode'] == 'virtual':
        #                 source.append(edge[1])
        print "Sub graph is done!"
        self.simplePVF(subG)
        #visited = self.traverse4PVF(subG, "bo%2", targetList)
        #for item in subG.nodes():
        #    if item not in visited:
        #        print "***"
        #       print item
        return subG

    def traverse4PVF(self, G, source, targetList):
        global aceBits
        visited = []
        visited.append(source)
        i = 0
        while i < len(visited):
            for edge in G.out_edges(visited[i]):
                if edge[1] not in visited:
                    visited.append(edge[1])
                    self.getParent(G, edge[1], visited)
            i += 1
            for target in targetList:
                if target in visited:
                    targetList.remove(target)
            if len(targetList) == 0:
                break
        return visited

    def getParent(self, G, node, visited):
        global aceBits
        opcode = ""
        oprandlist = []
        for edge in G.in_edges(node):
            if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                oprandlist.append(edge[0])
                opcode = G.edge[edge[0]][edge[1]]['opcode']
            #if edge[0] not in visited:
            #    if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
            #        opcode = G.edge[edge[0]][edge[1]]['opcode']
            if edge[0] not in visited:
                visited.append(edge[0])
                self.getParent(G, edge[0], visited)
        aceBits += self.instructionPVF(G, opcode, oprandlist, node)

    def getParent4CrashChain(self, G, node, max, min):
        global crashBits
        global stack

        global rangeList
        rangeList[node] = []
        rangeList[node].append(max)
        rangeList[node].append(min)
        oplist = []
        opcode = ""
        stack4recursion = []
        stack4recursion.append(node)
        while len(stack4recursion) != 0:
            node = stack4recursion.pop()
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oplist.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            if opcode != "load" and opcode != "store":
                dest = ""
                #if len(oplist) == 1:
                #    dest = G.successors(oplist[0])[0]
                #elif len(oplist) >= 2:
                #    snode = set(G.successors(oplist[0])).intersection(set(G.successors(oplist[1])))
                #    dest = snode.pop()
                #else:
                #    pass
                sorted_ops = sorted([i for i in oplist if node in self.indexMap[i]], key= lambda pos: self.indexMap[pos][node])
                if opcode != "" and len(sorted_ops) != 0:
                    #stack.extend(sorted_ops)
                    #stack.append(opcode)
                    sorted_ops.append(opcode)
                    self.calculateCrashChainBackward(G,sorted_ops)
                    #print sorted_ops
                    #print "crash chain"
            #if edge[0] not in visited:
            #    if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
            #        opcode = G.edge[edge[0]][edge[1]]['opcode']
                for op in sorted_ops:
                    if len(G.in_edges(op)) != 0:
                        #self.getParent4CrashChain(G, op, cbits)
                        stack4recursion.append(op)
            oplist = []

    def calculateCrashChain(self, G, opstack):
        manager = Manager()
        final = manager.list()
        #print counter
        if len(opstack) == 0:
            return []
        p = [ Process(target=self.worker, args=(G, opstack, final), name=i) for i in range(len(opstack))]
        for each in p:
            each.start()
        for each in p:
            each.join()
        return final

    def worker(self, G, opstack, final):
            if InstructionAbstraction.isint(current_process().name) == True:
                localop = opstack[int(current_process().name)]
                if localop not in config.memoryInst and localop not in config.bitwiseInst and localop not in config.computationInst and localop not in config.castInst and localop not in config.pointerInst and localop not in config.otherInst:
                    original = int(G.node[localop]['value'])
                    size = G.node[localop]['len']
                    for i in range(int(size)):
                        mask = (1 << i)
                        new = original^mask
                        mapping = {}
                        oplist = []
                        opcode = ""
                        for e in reversed(opstack):
                            if e not in config.memoryInst and e not in config.bitwiseInst and e not in config.computationInst and e not in config.castInst and e not in config.pointerInst and e not in config.otherInst:
                                oplist.append(e)
                            else:
                                opcode = e
                                #if opcode == "getelementptr":
                                #    print "hhhh"
                                v = self.brutalForce(G, oplist, opcode, mapping, localop, new)
                                node = ""
                                if len(oplist) == 1:
                                    node = G.successors(oplist[0])[0]
                                elif len(oplist) >= 2:
                                    snode = set(G.successors(oplist[0])).intersection(set(G.successors(oplist[1])))
                                    node = snode.pop()
                                else:
                                    pass
                                if localop == node:
                                    mapping[node] = new
                                else:
                                    mapping[node] = v
                                temp = mapping[node]
                                oplist = []
                                opcode = ""
                        final.append(temp)

    def brutalForce(self, G, replay, opcode, mapping, localop, new):

        values = []
        for op in replay:
            value = 0
            if op in mapping:
                value = mapping[op]
            else:
                value = int(G.node[op]['value'])
            if op == localop:
                value = new
            values.append(value)
            #for i in range(int(size)):
            #    mask = (1 << i)
            #    new_value = int(value)^mask
        ## sepcial case for ptr operations
        if opcode == "getelementptr":
            if len(replay) == 3:
                size = G.node[replay[0]]['realTy']
                values[1] = values[1]*int(size)
                #size = G.node[replay[2]]['len']
                if "structName" in G.node[replay[0]]:
                    structname = G.node[replay[0]]['structName']
                    sizelist = self.structMap["%"+structname]
                    t = 0
                    if values[2] >= len(sizelist):
                        for i in range(len(sizelist)):
                            t += sizelist[i]
                        t += (values[2]-len(sizelist) +1)*4
                    else:
                        for i in range(len(sizelist)):
                            t += sizelist[i]
                    values[2] = int(t/8)
                if "elementTy" in G.node[replay[0]]:
                    element = G.node[replay[0]]['elementTy']
                    values[2] = values[2]*int(int(element)/8)
            if len(replay) == 2:
                if "realTy" in G.node[replay[0]]:
                    size = G.node[replay[0]]['realTy']
                    values[1] = values[1]*int(int(size)/8)
        ret = self.calculateCrashInst(values, opcode)
        return ret


    def calculateCrashInst(self, values, opcode):
        #print opcode
        if opcode == "add" or opcode == "fadd":
            return sum(values)
        if opcode == "sub" or opcode == "fsub":
            assert(len(values) == 2)
            return values[0] - values[1]
        if opcode == "fmul" or opcode == "mul":
            assert(len(values) == 2)
            return values[0]*values[1]
        if opcode == "udiv"  or opcode == "sdiv" or opcode == "fdiv":
            assert(len(values) == 2)
            assert(values[1] != 0)
            return values[0]/values[1]
        if opcode == "sext":
            #bitstream = bin(values[0])
            #sign = ""
            #if bitstream.startswith("-"):
            #    bitstream = bitstream.lstrip("-")
            #    sign = "-"
            #if bitstream.startswith("0b"):
            #    bitstream = bitstream.lstrip("0b")
            #if len(bitstream) != config.OSbits:
            #    mis = 64 - len(bitstream)
            #    for i in range(mis):
             #       bitstream = '0'+bitstream
            #return int(sign+"0b"+bitstream, 2)
            return values[0]
        if opcode == "phi":
            return values[0]
        if opcode == "srem":
            assert(len(values) == 2)
            if values[1] == 0:
                return sys.maxint
            else:
                return values[0]%values[1]
        if opcode == "getelementptr":
            return sum(values)
        if opcode == "bitcast":
            #bitstream = bin(values[0])
            #sign = ""
            #if bitstream.startswith("-"):
            #    bitstream = bitstream.lstrip("-")
            #    sign = "-"
            #if bitstream.startswith("0b"):
            #    bitstream = bitstream.lstrip("0b")
            #if len(bitstream) != config.OSbits:
            #    mis = 64 - len(bitstream)
            #    for i in range(mis):
            #        bitstream = '0'+bitstream
            #return int(sign+"0b"+bitstream, 2)
            return values[0]

    def calculateCrashChainBackward(self, G, opstack):
        global rangeList
        #rangeList[node] = []
        #rangeList[node].append(max)
        #rangeList[node].append(min)
        oplist = []
        global finalBits
        for op in opstack:
            if op not in config.memoryInst and op not in config.bitwiseInst and op not in config.computationInst and op not in config.castInst and op not in config.pointerInst and op not in config.otherInst:
                oplist.append(op)
            else:
                opcode = op
                self.getRange4OPs(G, oplist, opcode, rangeList, finalBits)
                oplist = []
        return finalBits


    def getRange4OPs(self, G, oplist, opcode, rangeList, finalBits):

        node = ""
        if len(oplist) == 1:
            nlist = G.successors(oplist[0])
            for n in nlist:
                node = n
                if node in rangeList:
                    break
        else:
            snode = set(G.successors(oplist[0])).intersection(set(G.successors(oplist[1])))
            while len(snode) != 0:
                node = snode.pop()
                if node in rangeList:
                    break
        if node not in rangeList:
            print "here"
        max_range = int(rangeList[node][0])
        min_range = int(rangeList[node][1])
        if opcode == "add" or opcode == "fadd":
            assert(len(oplist) == 2)
            for i in range(2):
                    max_op = max_range - int(G.node[oplist[1-i]]['value'])
                    min_op = min_range - int(G.node[oplist[1-i]]['value'])
                    if oplist[i] not in rangeList:
                        rangeList[oplist[i]] = []
                        rangeList[oplist[i]].append(max_op)
                        rangeList[oplist[i]].append(min_op)
                    type = G.node[oplist[i]]['len']
                    #if "constant" not in oplist[i]:
                    finalBits.append(self.checkRange(G,oplist[i],max_op, min_op, type))

        if opcode == "sub" or opcode == "fsub":
            assert(len(oplist) == 2)
            # the first operand
            max_op = max_range + int(G.node[oplist[1]]['value'])
            min_op = min_range + int(G.node[oplist[1]]['value'])
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            #if "constant" not in oplist[0]:
            finalBits.append(self.checkRange(G, oplist[0],max_op, min_op, type))
            max_op = int(G.node[oplist[0]]['value']) - min_range
            min_op = int(G.node[oplist[0]]['value']) - max_range
            if oplist[1] not in rangeList:
                rangeList[oplist[1]] = []
                rangeList[oplist[1]].append(max_op)
                rangeList[oplist[1]].append(min_op)
            type = G.node[oplist[1]]['len']
            #if "constant" not in oplist[1]:
            finalBits.append(self.checkRange(G, oplist[1],max_op, min_op, type))

        if opcode == "fmul" or opcode == "mul":
            assert(len(oplist) == 2)
            for i in range(2):
                    if int(G.node[oplist[1-i]]['value']) == 0:
                        max_op = sys.maxint
                        min_op = -sys.maxint-1
                    else:
                        max_op = max_range/int(G.node[oplist[1-i]]['value'])
                        min_op = min_range/int(G.node[oplist[1-i]]['value'])
                    if oplist[i] not in rangeList:
                        rangeList[oplist[i]] = []
                        rangeList[oplist[i]].append(max_op)
                        rangeList[oplist[i]].append(min_op)
                    type = G.node[oplist[i]]['len']
                    #if "constant" not in oplist[i]:
                    finalBits.append(self.checkRange(G,oplist[i],max_op, min_op, type))

        if opcode == "udiv"  or opcode == "sdiv" or opcode == "fdiv":
            assert(len(oplist) == 2)
            # the first operand
            max_op = max_range*int(G.node[oplist[1]]['value'])
            min_op = min_range*int(G.node[oplist[1]]['value'])
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            #if "constant" not in oplist[0]:
            finalBits.append(self.checkRange(G, oplist[0] ,max_op, min_op, type))
            max_op = int(G.node[oplist[0]]['value'])/min_range
            min_op = int(G.node[oplist[0]]['value'])/max_range
            if oplist[1] not in rangeList:
                rangeList[oplist[1]] = []
                rangeList[oplist[1]].append(max_op)
                rangeList[oplist[1]].append(min_op)
            type = G.node[oplist[1]]['len']
            #if "constant" not in oplist[1]:
            finalBits.append(self.checkRange(G, oplist[1] ,max_op, min_op, type))

        if opcode == "sext":
            #bitstream = bin(values[0])
            #sign = ""
            #if bitstream.startswith("-"):
            #    bitstream = bitstream.lstrip("-")
            #    sign = "-"
            #if bitstream.startswith("0b"):
            #    bitstream = bitstream.lstrip("0b")
            #if len(bitstream) != config.OSbits:
            #    mis = 64 - len(bitstream)
            #    for i in range(mis):
             #       bitstream = '0'+bitstream
            #return int(sign+"0b"+bitstream, 2)
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            finalBits.append(self.checkRange(G, oplist[0],max_op, min_op, type))

        if opcode == "phi":
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            finalBits.append(self.checkRange(G, oplist[0] ,max_op, min_op, type))

        if opcode == "trunc":
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            finalBits.append(self.checkRange(G, oplist[0] ,max_op, min_op, type))

        if opcode == "srem":
            assert(len(oplist) == 2)
            type = int(G.node[oplist[0]]['len'])
            value = int(G.node[oplist[0]]['value'])
            tmp = []
            for i in range(type):
                mask = (1 << i)
                value = value ^ mask
                if value%int(G.node[oplist[1]]['value']) > max or value%int(G.node[oplist[1]]['value']) < min:
                    tmp.append(value)
            rangeList[oplist[0]] = []
            rangeList[oplist[0]].append(max(tmp))
            rangeList[oplist[0]].append(min(tmp))
            type = int(G.node[oplist[1]]['len'])
            value = int(G.node[oplist[1]]['value'])
            tmp = []
            for i in range(type):
                mask = (1 << i)
                value = value ^ mask
                if value == 0:
                    continue
                if int(G.node[oplist[1]]['value'])%value > max or int(G.node[oplist[1]]['value'])%value < min:
                    tmp.append(value)
            rangeList[oplist[1]] = []
            rangeList[oplist[1]].append(max(tmp))
            rangeList[oplist[1]].append(min(tmp))

        if opcode == "getelementptr":
            if len(oplist) == 2:
                if "realTy" in G.node[oplist[0]]:
                    size = G.node[oplist[0]]['realTy']
                    #values[1] = values[1]*int(int(size)/8)
                    max_op = max_range - int(G.node[oplist[1]]['value'])*int(size)/8
                    min_op = min_range - int(G.node[oplist[1]]['value'])*int(size)/8
                    if oplist[0] not in rangeList:
                        rangeList[oplist[0]] = []
                        rangeList[oplist[0]].append(max_op)
                        rangeList[oplist[0]].append(min_op)
                    type = G.node[oplist[0]]['len']
                    #if "constant" not in oplist[0]:
                    finalBits.append(self.checkRange(G, oplist[0],max_op, min_op, type))
                    max_op = (max_range - int(G.node[oplist[0]]['value']))*8/int(size)
                    min_op = (min_range - int(G.node[oplist[0]]['value']))*8/int(size)
                    if oplist[1] not in rangeList:
                        rangeList[oplist[1]] = []
                        rangeList[oplist[1]].append(max_op)
                        rangeList[oplist[1]].append(min_op)
                    type = G.node[oplist[1]]['len']
                    #if "constant" not in oplist[1]:
                    finalBits.append(self.checkRange(G, oplist[1] ,max_op, min_op, type))

            if len(oplist) == 3:
                if "realTy" in G.node[oplist[0]]:

                    size = G.node[oplist[0]]['realTy']
                #size = G.node[replay[2]]['len']
                    if "structName" in G.node[oplist[0]]:
                        structname = G.node[oplist[0]]['structName']
                        sizelist = self.structMap["%"+structname]
                        t = 0
                        if int(G.node[oplist[2]]['value']) >= len(sizelist):
                            for i in range(len(sizelist)):
                                t += sizelist[i]
                            t += (int(G.node[oplist[2]]['value'])-len(sizelist) +1)*4
                        else:
                            for i in range(int(G.node[oplist[2]]['value'])):
                                t += sizelist[i]
                        max_op = max_range - int(G.node[oplist[1]]['value'])*int(size)/8 - int(t/8)
                        min_op = min_range - int(G.node[oplist[1]]['value'])*int(size)/8 - int(t/8)
                        if oplist[0] not in rangeList:
                            rangeList[oplist[0]] = []
                            rangeList[oplist[0]].append(max_op)
                            rangeList[oplist[0]].append(min_op)
                        type = G.node[oplist[0]]['len']
                        #if "constant" not in oplist[0]:
                        finalBits.append(self.checkRange(G, oplist[0] ,max_op, min_op, type))
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(t/8))*8/int(size)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(t/8))*8/int(size)
                        if oplist[1] not in rangeList:
                            rangeList[oplist[1]] = []
                            rangeList[oplist[1]].append(max_op)
                            rangeList[oplist[1]].append(min_op)
                        type = G.node[oplist[1]]['len']
                        #if "constant" not in oplist[1]:
                        finalBits.append(self.checkRange(G, oplist[1],max_op, min_op, type))
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/8
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/8
                        if oplist[2] not in rangeList:
                            rangeList[oplist[2]] = []
                            rangeList[oplist[2]].append(max_op)
                            rangeList[oplist[2]].append(min_op)
                        type = G.node[oplist[2]]['len']
                        #if "constant" not in oplist[2]:
                        finalBits.append(self.checkRange(G, oplist[2],max_op, min_op, type))

                    if "elementTy" in G.node[oplist[0]]:
                        element = G.node[oplist[0]]['elementTy']
                        t = int(G.node[oplist[2]]['value'])*int(int(element)/8)
                        max_op = max_range - int(G.node[oplist[1]]['value'])*int(size)/8 - t
                        min_op = min_range - int(G.node[oplist[1]]['value'])*int(size)/8 - t
                        if oplist[0] not in rangeList:
                            rangeList[oplist[0]] = []
                            rangeList[oplist[0]].append(max_op)
                            rangeList[oplist[0]].append(min_op)
                        type = G.node[oplist[0]]['len']
                        #if "constant" not in oplist[0]:
                        finalBits.append(self.checkRange(G, oplist[0],max_op, min_op, type))
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - t)/int(size)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - t)/int(size)
                        if oplist[1] not in rangeList:
                            rangeList[oplist[1]] = []
                            rangeList[oplist[1]].append(max_op)
                            rangeList[oplist[1]].append(min_op)
                        type = G.node[oplist[1]]['len']
                        #if "constant" not in oplist[1]:
                        finalBits.append(self.checkRange(G,oplist[1],max_op, min_op, type))
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/int(int(element)/8)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/int(int(element)/8)
                        if oplist[2] not in rangeList:
                            rangeList[oplist[2]] = []
                            rangeList[oplist[2]].append(max_op)
                            rangeList[oplist[2]].append(min_op)
                        type = G.node[oplist[2]]['len']
                        #if "constant" not in oplist[2]:
                        finalBits.append(self.checkRange(G, oplist[2],max_op, min_op, type))



        if opcode == "bitcast":
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            finalBits.append(self.checkRange(G, oplist[0], max_op, min_op, type))

        if opcode == "alloca":
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            finalBits.append(self.checkRange(G, oplist[0], max_op, min_op, type))



    def checkRange(self,G, node, max_f, min_f, type):
        """

        :rtype : integer
        """
        global rangeList
        global gcount
        #bitstream = bin(min)
        #if bitstream.startswith("0b"):
        #    bitstream = bitstream.lstrip("0b")
        #if len(bitstream) != config.OSbits:
        #    mis = 64 - len(bitstream)
        #    for i in range(mis):
        #        bitstream = '0'+bitstream
        #assert(len(bitstream) == config.OSbits)
        #flag = 0
        #count = 0
        #for pos, bit in enumerate(bitstream):
        #    if flag == 1:
        #       if bit == "1":
        #            count += 1
        #    if bit == "1" and flag == 0:
        #        flag = 1
        #        count += pos
        #return
        counter = 0
        address = int(G.node[node]['value'])
        for i in range(type):
            mask = (1 << i)
            new = address
            new = new ^ mask
            if new > max_f or new < min_f:
                if len(rangeList[node]) == 2:
                    rangeList[node].append([])
                    rangeList[node][2].append(i)
                    counter += 1
                else:
                    if i not in rangeList[node][2]:
                       rangeList[node][2].append(i)
                    counter += 1
        return counter


    def checkRange1(self,address, max_f, min_f, esp, type):
        """

        :rtype : integer
        """
        counter = 0
        min_new = 0
        for i in range(type):
            mask = (1 << i)
            new = address
            new = new ^ mask
            if new > max_f:
                counter += 1
            if new < min_f:
                if esp == 0:
                    counter += 1
                    min_new = min_f
                else:
                    if new < esp - 65536 - 128:
                        counter += 1
                        min_new = min(esp - 65536 - 128, min_f)
        return [counter, min_new]

    def instructionPVF(self, G, opcode, oplist, node):
        global crashBits
        global stack
        bb = 0
        removed = 0
        counter = 0
        if opcode in config.computationInst:
            for op in oplist:
                #if "constant" in op:
                #    continue
                res = G.node[op]['len']
                bb += int(res)
            if opcode == "mul" or opcode == "fmul":
                    for index, op in enumerate(oplist):
                        if (G.node[op]['value'] == int(0) or G.node[op]['value'] == float(0)):
                           bb -= G.node[oplist[1-index]]['len']
            #res = G.node[node]['len']
            #b += int(res)
        elif opcode in config.bitwiseInst:
            if opcode == "and":
                 res = re.findall('\d+', G.node[oplist[0]]['len'])
                 size = int(res[0])
                 base = pow(2,size)-1
                 for op in oplist:
                     value = int(G.node[op]['bits'])
                     base &= value
                 bb += bin(base).count("0")
            if opcode == "or":
                 base = 0
                 for op in oplist:
                     value = int(G.node[op]['bits'])
                     base |= value
                     bb += bin(base).count("1")
            if opcode == "shl" or opcode == "lshr" or opcode == "ashr":
                size = int(G.node[oplist[0]]['len'])
                shift = int(G.node[oplist[1]['value']])
                bb += size - shift
            #     pass
            #for op in oplist:
            #    res = G.node[op]['len']
            #    bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        elif opcode in config.pointerInst:
            for op in oplist:
                #if "constant" in op:
                #    continue
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']

            #b += int(res)
        elif opcode in config.memoryInst:
            if opcode == "load":
                res = 0
                for op in oplist:
                    if "mem" in G.node[op]:
                        index = G.node[op]['mem']
                        range_mem = self.memory[index]
                        address = op.split("+")[0]
                        res = G.node[op]['len']
                        type = int(res)
                        bb += res
                        max = 0
                        min = 0
                        removed1 = 0
                        start_node = ""
                        #for item1, item2, item3 in grouper(3, range_mem):
                        #print range_mem
                        #print address
                        #print "+++++++"
                        for key in range_mem.keys():
                            if key == 'heap':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 'stack':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                esp = range_mem['esp'][0]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),int(esp), type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 0:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 1:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                        if max == 0 or min == 0:
                            print "wrong!!!!"
                        for edge in G.in_edges(op):
                            if G.edge[edge[0]][edge[1]]['opcode'] == "virtual":
                                    if "pre" in G.node[node]:
                                        if edge[0] == G.node[node]['pre']:
                                            start_node = edge[0]
                                            self.getParent4CrashChain(G, edge[0], max, min)
                                            #self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                            #print "one ld/st is done"
                                            break
                                   #if G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] in config.pointerInst or G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] == "alloca":
                        #if len(stack) != 0:
                            #final = self.calculateCrashChain(G, stack)
                            #print len(final)
                        #    final = self.calculateCrashChainBackward(G, stack, max, min, start_node)
                            #for item in final:
                            #    removed += item
                            # the memory instruciton itself

                            #print "++++++++"
                            #print removed
                            #print removed1
                        #else:
                        for edge in G.in_edges(op):
                                # for alloca
                            if "root" in edge[0]:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                break
                            if "mem" in G.node[edge[0]]:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                removed += removed1
                                break
                            if edge[0] in config.Outbound:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                removed += removed1
                                break
                        if len(G.in_edges(op)) == 0:
                            self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                        removed += removed1
                        stack = []

            if opcode == "store":
                if "mem" in G.node[node]:
                    index = G.node[node]['mem']
                    range_mem = self.memory[index]
                    address = node.split("+")[0]
                    res = G.node[node]['len']
                    type = int(res)
                    bb += type
                    max = 0
                    min = 0
                    removed1 = 0
                    start_node = ""
                    for key in range_mem.keys():
                            if key == 'heap':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 0:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 1:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 'stack':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                esp = range_mem['esp'][0]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min] = self.checkRange1(int(address), int(item2), int(item1),int(esp), type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break

                    for edge in G.in_edges(node):
                            if G.edge[edge[0]][edge[1]]['opcode'] == "virtual":
                                if "pre" in G.node[edge[0]]:
                                    for op in oplist:
                                        if op in G.node[edge[0]]['pre']:
                                    #if G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] in config.pointerInst or G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] == "alloca":
                                    #     if flag == 0:
                                    #       removed += removed
                                    #        flag = 1
                                    #        break
                                            #bb += int(G.node[op]['len'])
                                            start_node = edge[0]
                                            self.getParent4CrashChain(G, edge[0],max, min)
                                            #self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                        #print "one ld/st is done"
                                            break
                                        #start_node = edge[0]
                    #if len(stack) != 0:
                    #        final = self.calculateCrashChainBackward(G, stack, max, min, start_node)
                            #for item in final:
                            #    removed += item
                            #removed += removed1
                            #print "++++++++"
                            #print removed
                            #print removed1
                    #else:
                    for edge in G.in_edges(node):
                                # for alloca
                            if "root" in edge[0]:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                break
                            if "mem" in G.node[edge[0]]:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                #removed += removed1
                                break
                            if edge[0] in config.Outbound:
                                self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                #removed += removed1
                                break
                    if len(G.in_edges(node)) == 0:
                        self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    removed += removed1
                    #self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    stack = []

        elif opcode in config.castInst:
            for op in oplist:
                #if "constant" in op:
                #    continue
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        elif opcode in config.otherInst:
            if opcode == "icmp" or opcode == "fcmp":
                predicate = G.edge[oplist[0]][node]['pred']
                bb += self.checkCMPRange(G, oplist, predicate)
            else:
                if opcode == "phi":
                    pass
                else:
                    for op in oplist:
                    #if "constant" in op:
                    #    continue
                        res = G.node[op]['len']
                        bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        else:
            print opcode
            print "WRONG"
        #if opcode not in config.memoryInst:
        #    bb += G.node[node]['len']
        bb -= removed
        #print b
        #print "######"
        #print node
        #print oplist
        #print counter
        return bb

    def checkCMPRange(self,G,oplist,predicate):
        count_1 = 0
        count_2 = 0
        if int(predicate) == 40:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new < value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 < new:
                    count_2 += 1
        if int(predicate) == 41:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new <= value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 <= new:
                    count_2 += 1
        if int(predicate) == 39:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new >= value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 >= new:
                    count_2 += 1
        if int(predicate) == 32:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new == value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 == new:
                    count_2 += 1
        if int(predicate) == 33:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new != value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 != new:
                    count_2 += 1
        if int(predicate) == 38:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if new >= value2:
                    count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if value1 >= new:
                    count_2 += 1

        oplist_new = []
        opcode = ""
        stack4recursion = []
        stack4recursion.extend(oplist)
        icmpbits = {}
        icmpbits[count_1] = []
        icmpbits[count_1].append(oplist[0])
        icmpbits[count_2] = []
        icmpbits[count_2].append(oplist[1])
        final = count_1+count_2
        while len(stack4recursion) != 0:
            node = stack4recursion.pop()
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oplist_new.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            if opcode != "load" and opcode != "load":
                for op in oplist_new:
                        #self.getParent4CrashChain(G, op, cbits)
                        stack4recursion.append(op)
                        if node in icmpbits[count_1]:
                            icmpbits[count_1].append(op)
                        if node in icmpbits[count_2]:
                            icmpbits[count_2].append(op)
            else:
                if opcode == "load":
                    for op in oplist_new:
                        for edge in G.in_edges(op):
                            opcode_new = G.edge[edge[0]][edge[1]]['opcode']
                            if opcode_new == "store":
                                stack4recursion.append(edge[1])
                        if node in icmpbits[count_1]:
                            icmpbits[count_1].append(edge[1])
                        if node in icmpbits[count_2]:
                            icmpbits[count_2].append(edge[1])
            oplist_new = []
        final -= count_1*(len(icmpbits[count_1]))
        final -= count_2*(len(icmpbits[count_2]))
        return final

    def simplePVF(self,G):
        global rangeList
        global finalBits
        b = 0
        print "Total subG"
        print len(G.nodes())
        count = 0
        for node in G.nodes_iter():
            #if "pre" in G.node[node]:
                #print G.node[node]['pre']
                #print "###"
            oprandlist = []
            opcode = ""
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oprandlist.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            if opcode != "":
                #print "======"
                #print opcode
                #print oprandlist
                b += self.instructionPVF(G, opcode, oprandlist, node)
                count += 1
        crash = 0
        print count
        for i in rangeList:
            if len(rangeList[i]) == 3:
                crash += len(rangeList[i][2])
        print crash
        crash = 0
        for i in finalBits:
            crash += i
        #print "without cmp"
        #print b
        #for node in self.G.nodes_iter():
        #    if node in G:
        #        continue
        #    oprandlist = []
        #    opcode = ""
        #    for edge in self.G.in_edges(node):
        #        if "virtual" not in self.G.edge[edge[0]][edge[1]]['opcode']:
        #            oprandlist.append(edge[0])
        #            opcode = self.G.edge[edge[0]][edge[1]]['opcode']
        #    if opcode == "icmp" or opcode == "fcmp":
        #        b += self.instructionPVF(self.G, opcode, oprandlist, node)
        print "Crash:"
        print crash
        print "SDC bits"
        print b