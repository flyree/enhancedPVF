import re
import sys
import os
import random
import networkx as nx
import copy
from itertools import izip_longest
import setting as config

aceBits = 0
crashBits = 0
stack = []

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class PVF:
    def __init__(self, G, trace):
        self.G = G
        assert(len(trace) == 3)
        self.trace = trace[0]
        self.remap = trace[1]
        self.memory = trace[2]


    def computePVF(self, targetList):

        #----------------
        # Get the predecessors of the target node
        #----------------
        global aceBits
        global crashBits
        predecessors = []
        for node in self.G.nodes():
            for target in targetList:
                if target in node:
                    predecessors.append(node)
        for node in predecessors[:]:
            for edge in self.G.out_edges(node):
                if self.G.edge[edge[0]][edge[1]]['opcode'] != 'virtual':
                    predecessors.remove(node)
        targetList = list(predecessors)
        i = 0
        while i < len(predecessors):
            flag = 0
            newlist = self.G.predecessors(predecessors[i])
            offsets = {}
            for newnode in newlist:
               if 'dest' in self.G.node[newnode].keys():
                   if self.G.node[newnode]['dest'] in predecessors:
                       flag = 1
                       offset = predecessors.index(self.G.node[newnode]['dest'])
                       if offset not in offsets.keys():
                           offsets[offset] = []
                           offsets[offset].append(newnode)
                       else:
                           offsets[offset].append(newnode)
               else:
                   if newnode not in predecessors:
                       predecessors.append(newnode)
            if flag == 1:
                for node in offsets[sorted(offsets)[0]]:
                    if node not in predecessors:
                        predecessors.append(node)
            i += 1
        subG = self.G.subgraph(predecessors)
        source = []
        for node in subG:
            if subG.in_degree(node) == 0:
                source.append(node)
            if subG.in_degree(node) == 1:
                for edge in subG.in_edges(node):
                    if subG.edge[edge[0]][edge[1]]['opcode'] == 'virtual':
                        source.append(edge[1])
        print len(subG.nodes())
        counter = 0
        for e in subG.edges_iter():
            if subG.edge[e[0]][e[1]]['opcode'] in config.pointerInst:
                counter += 1
        print counter
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

    def getParent4CrashChain(self, G, node, cbits):
        global crashBits
        global stack
        oplist = []
        opcode = ""
        for edge in G.in_edges(node):
            if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                oplist.append(edge[0])
                opcode = G.edge[edge[0]][edge[1]]['opcode']
        if opcode != "load" and opcode != "store":
            sorted_ops = sorted([i for i in oplist if "index" in G.node[i].keys()], key= lambda pos: G.node[pos]['index'], reverse = True)
            if opcode != "" and len(sorted_ops) != 0:
                stack.append(opcode)
                stack.extend(sorted_ops)
            #if edge[0] not in visited:
            #    if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
            #        opcode = G.edge[edge[0]][edge[1]]['opcode']
            for op in sorted_ops:
                if len(G.in_edges(op)) != 0:
                    self.getParent4CrashChain(G, op, cbits)

    def calculateCrashChain(self, G, opstack):
        final = []
        temp = 0
        localstack = opstack
        popstack = copy.deepcopy(opstack)
        counter = 0
        for localop in localstack:
            if localop not in config.memoryInst and localop not in config.bitwiseInst and localop not in config.computationInst and localop not in config.castInst and localop not in config.pointerInst and localop not in config.otherInst:
                original = int(G.node[localop]['value'])
                size = G.node[localop]['len']
                for i in range(int(size)):
                    mask = (1 << i)
                    new = original^mask
                    mapping = {}
                    oplist = []
                    opcode = ""
                    while len(popstack) != 0:
                        e = popstack.pop()
                        if e not in config.memoryInst and e not in config.bitwiseInst and e not in config.computationInst and e not in config.castInst and e not in config.pointerInst and e not in config.otherInst:
                            oplist.append(e)
                        else:
                            opcode = e
                            if opcode == "phi" and len(oplist) == 0:
                                print "hhhh"
                            v = self.brutalForce(G, oplist, opcode, mapping, localop, new)
                            node = G.successors(oplist[0])[0]
                            mapping[node] = v
                            temp = v
                            oplist = []
                            opcode = ""
                            counter += 1
                    final.append(temp)
                    popstack = copy.deepcopy(opstack)
        #print counter
        print len(final)
        return final


    def brutalForce(self, G, replay, opcode, mapping, localop, new):

        values = []
        for op in replay:
            value = 0
            if op in mapping.keys():
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
                values[2] = values[2]*int(config.gepsize)
            if len(replay) == 2:
                #size = G.node[replay[1]]['len']
                values[1] = values[1]*int(config.gepsize)
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
            bitstream = bin(values[0])
            sign = ""
            if bitstream.startswith("-"):
                bitstream = bitstream.lstrip("-")
                sign = "-"
            if bitstream.startswith("0b"):
                bitstream = bitstream.lstrip("0b")
            if len(bitstream) != config.OSbits:
                mis = 64 - len(bitstream)
                for i in range(mis):
                    bitstream = '0'+bitstream
            return int(sign+"0b"+bitstream, 2)
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
            bitstream = bin(values[0])
            sign = ""
            if bitstream.startswith("-"):
                bitstream = bitstream.lstrip("-")
                sign = "-"
            if bitstream.startswith("0b"):
                bitstream = bitstream.lstrip("0b")
            if len(bitstream) != config.OSbits:
                mis = 64 - len(bitstream)
                for i in range(mis):
                    bitstream = '0'+bitstream
            return int(sign+"0b"+bitstream, 2)






    def checkRange(self, address, max, min, type):
        """

        :rtype : integer
        """
        bitstream = bin(min)
        if bitstream.startswith("0b"):
            bitstream = bitstream.lstrip("0b")
        if len(bitstream) != config.OSbits:
            mis = 64 - len(bitstream)
            for i in range(mis):
                bitstream = '0'+bitstream
        assert(len(bitstream) == config.OSbits)
        flag = 0
        count = 0
        for pos, bit in enumerate(bitstream):
            if flag == 1:
                if bit == "1":
                    count += 1
            if bit == "1" and flag == 0:
                flag = 1
                count += pos
        return count

    def instructionPVF(self, G, opcode, oplist, node):
        global crashBits
        global stack
        bb = 0
        removed = 0
        counter = 0
        if opcode in config.computationInst:
            for op in oplist:
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        if opcode in config.bitwiseInst:
            # if opcode == "and":
            #     res = re.findall('\d+', G.node[oplist[0]]['len'])
            #     size = int(res[0])
            #     base = pow(2,size)-1
            #     for op in oplist:
            #         value = int(G.node[op]['bits'])
            #         base &= value
            #     b = bin(base).count("0")
            # if opcode == "or":
            #     base = 0
            #     for op in oplist:
            #         value = int(G.node[op]['bits'])
            #         base |= value
            #         b = bin(base).count("1")
            # if opcode == "shl" or opcode == "lshr" or opcode == "ashr":
            #     pass
            for op in oplist:
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        if opcode in config.pointerInst:
            for op in oplist:
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']

            #b += int(res)
        if opcode in config.memoryInst:
            if opcode == "load":
                res = 0
                for op in oplist:
                    if "mem" in G.node[op].keys():
                        index = G.node[op]['mem']
                        range_mem = self.memory[index]
                        address = op.split("+")[0]
                        res = G.node[op]['len']
                        type = int(res)
                        max = 0
                        min = 0
                        for item1, item2 in grouper(2, range_mem):
                            if int(item1) <= int(address) and int(address) <= int(item2):
                                removed1 = self.checkRange(int(address), int(item2), int(item1), type)
                                max = int(item2)
                                min = int(item1)
                                print removed1
                                break
                        for edge in G.in_edges(op):
                            if G.edge[edge[0]][edge[1]]['opcode'] == "virtual":
                                    if "pre" in G.node[node].keys():
                                        if edge[0] == G.node[node]['pre']:
                                            self.getParent4CrashChain(G, edge[0], removed)
                                            break
                                   #if G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] in config.pointerInst or G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] == "alloca":
                        if len(stack) != 0:
                            final = self.calculateCrashChain(G, stack)
                            for item in final:
                                if item > max or item < min:
                                    removed += 1
                        else:
                            for edge in G.in_edges(address):
                                if "root" in edge[0]:
                                    removed += removed1
                                    break
                                if edge[0] in config.Outbound:
                                    removed += removed1
                                    break
                        #counter += 1
                        if removed > 10:
                            print final
                            print max
                            print min
                            print stack
                        stack = []

                        print removed

                        print "####"
                        #    else:
                        #        if G.edge[edge[0]][edge[1]]['opcode'] in config.pointerInst:
                        #            removed *= 2
                        #            break
                        #    if flag == 1:
                        #        break
            if opcode == "store":
                if "mem" in G.node[node].keys():
                    index = G.node[node]['mem']
                    range_mem = self.memory[index]
                    address = node.split("+")[0]
                    res = G.node[node]['len']
                    type = int(res)
                    flag = 0
                    max = 0
                    min = 0
                    for item1, item2 in grouper(2, range_mem):
                        if int(item1) <= int(address) and int(address) <= int(item2):
                            removed1 = self.checkRange(int(address), int(item2), int(item1), type)
                            max = int(item2)
                            min = int(item1)
                            print removed1
                            break
                    for edge in G.in_edges(node):
                            if G.edge[edge[0]][edge[1]]['opcode'] == "virtual":
                                    #if G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] in config.pointerInst or G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] == "alloca":
                                    #     if flag == 0:
                                    #       removed += removed
                                    #        flag = 1
                                    #        break
                                self.getParent4CrashChain(G, edge[0],removed)
                    if len(stack) != 0:
                            final = self.calculateCrashChain(G, stack)
                            for item in final:
                                if item > max or item < min:
                                    removed += 1
                    else:
                        for edge in G.in_edges(address):
                            if "root" in edge[0]:
                                removed += removed1
                                break
                            if edge[0] in config.Outbound:
                                removed += removed1
                                break
                    #counter += 1
                    if removed > 10:
                            print final
                            print max
                            print min
                            print stack
                    print removed
                    print "####"
                    stack = []
                    #        else:
                    #            if G.edge[edge[0]][edge[1]]['opcode'] in config.pointerInst:
                    #                removed *= 2
                    #                break
                    #        if flag == 1:
                    #            break
            #removed = 0
        if opcode in config.castInst:
            for op in oplist:
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        if opcode in config.otherInst:
            for op in oplist:
                res = G.node[op]['len']
                bb += int(res)
            #res = G.node[node]['len']
            #b += int(res)
        bb -= removed
        #print b
        #print "######"
        #print node
        #print oplist
        #print counter
        return bb

    def simplePVF(self,G):
        b = 0
        for node in G.nodes_iter():
            #if "pre" in G.node[node].keys():
                #print G.node[node]['pre']
                #print "###"
            oprandlist = []
            opcode = ""
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oprandlist.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            b += self.instructionPVF(G, opcode, oprandlist, node)
        print "SDC bits"
        print b