import re
import sys
import os
import random
import networkx as nx
import copy
from itertools import izip_longest
import setting as config
import InstructionAbstraction
from sets import Set
from multiprocessing import Manager, Process, current_process, Queue
import math
import time
import collections
from collections import OrderedDict
aceBits = 0
crashBits = 0
stack = []
rangeList = {}
gcount = 0
finalBits = []
loadstore = []
finalBits_control = []
control_start = 0
lower_bound = 1
f_level = 1
loadstore_bits = {}

pop = 0.2
K = 1

ranking = {}

def random_subset(iterator, K):
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len(result) == 0:
            result.append(item)
        if len(iterator)/len(result) > K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < len(result):
                result[s] = item

    return result

def ordered_subset(iterator,K):
    unordered = {}
    noplus = []
    count = 0
    serialnumber = 0
    for item in iterator:
        serialnumber = count
        if "constant" in item:
            res = re.findall('constant\d+',item)
            serialnumber = res[0].split("constant")[1]
            unordered[serialnumber] = iterator.index(item)
        else:
            if "+" in item:
                serialnumber = int(item.split("+")[1])
                unordered[serialnumber] = iterator.index(item)
            else:
                noplus.append(item)
    if len(unordered.keys()) == 1:
        for item in iterator:
            serialnumber = int(item)
            unordered[serialnumber] = iterator.index(item)
    ordered = collections.OrderedDict(sorted(unordered.items()))
    klist = []
    klist.extend(noplus)
    print klist
    for k,v in ordered.iteritems():
        klist.append(iterator[v])
        if len(klist) > len(iterator)*K:
            break
    print klist
    return klist




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

def iszero(x):
    try:
        if isfloat(x):
            x = float(x)
            if x == float(0):
                return True
            else:
                return False
        if isint(x):
            x = int(x)
            if x == 0:
                return True
            else:
                return False
    except ValueError:
        return False


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class PVF:
    def __init__(self, G, trace, indexMap,global_hash_cycle, cycle_index_lookup):
        self.G = G
        assert(len(trace) == 3)
        self.trace = trace[0]
        self.remap = trace[1]
        self.memory = trace[2]
        fm = InstructionAbstraction.FunctionMapping(config.IRpath)
        structMap = fm.extractStruct()
        self.structMap = structMap
        self.indexMap = indexMap
        self.global_hash_cycle = global_hash_cycle
        self.cycle_index_lookup = cycle_index_lookup

    def ordered_subset_bycycle(self, iterator,K):
        unordered = {}
        noplus = []
        count = 0
        serialnumber = 0
        for item in iterator:
            index = int(self.G.node[item]['cycle'])
            if index not in unordered:
                unordered[index] = iterator.index(item)
            else:
                index += random.random()
                unordered[index] = iterator.index(item)
        ordered = collections.OrderedDict(sorted(unordered.items()))
        klist = []
        for k,v in ordered.iteritems():
            klist.append(iterator[v])
            if len(klist) >= len(iterator)*K:
                break
        return klist


    def bfs(self, G, source):
        bfs_nodes = []
        bfs_nodes.extend(source)
        visited = Set()
        gl_pre = []
        i = 0
        while i < len(bfs_nodes):
            if "mem" not in G.node[bfs_nodes[i]]:
                for edge in G.out_edges(bfs_nodes[i]):
                    if edge[1] not in visited:
                        bfs_nodes.append(edge[1])
                if "pre" in G.node[bfs_nodes[i]]:
                    #gl_pre.append(G.node[bfs_nodes[i]]['pre'])
                    bfs_nodes.append(G.node[bfs_nodes[i]]['pre'])
                visited.add(bfs_nodes[i])
            else:
                #for edge in G.out_edges(bfs_nodes[i]):
                #    if "store" in G.edge[edge[0]][edge[1]]['opcode']:
                #        if edge[1] not in visited:
                #            bfs_nodes.append(edge[1])
                #        if G.node[edge[1]]['post'] not in visited:
                #            bfs_nodes.append(G.node[edge[1]]['post'])
                #    elif edge[1] in gl_pre:
                #        bfs_nodes.append(edge[1])
                #        gl_pre.remove(edge[1])
                if "store" in G.node[bfs_nodes[i]]:
                    op = G.node[bfs_nodes[i]]['store']
                    if op not in visited:
                        bfs_nodes.append(op)
                    if G.node[op]['post'] not in visited:
                        bfs_nodes.append(G.node[op]['post'])
                #for node in gl_pre:
                #        if G.has_edge(node,bfs_nodes[i]):
                #            bfs_nodes.append(node)
                #            gl_pre.remove(node)
                #            break

            i += 1
        return bfs_nodes

    def computeCrashRate(self):
        for node in self.G.nodes_iter():
            numOut = 0
            for edge in self.G.out_edges(node):
                if "virtual" not in self.G.edge[edge[0]][edge[1]]['opcode']:
                    numOut += 1

            #if numOut == 0:
            #    print node
            #    for edge in uG.in_edges(node):
            #            if "alloca" == uG.edge[edge[0]][edge[1]]['opcode']:
            #                numOut += 1
            self.G.node[node]['out_edge'] = numOut
        self.simplePVF(self.G,self.G)

    def computePVF(self, targetList):

        #----------------
        # Get the predecessors of the target node
        #----------------
        global aceBits
        global crashBits
        global lower_bound
        global K
        predecessors = []
        predecessors_control = []
        predecessors_memory = []
        for node in self.G:
            oprandlist = []
            opcode = ""
            for target in targetList:
                if target in node:
                    #predecessors.append(node)
                    for edge in self.G.out_edges(node):
                        if "virtual" in self.G.edge[edge[0]][edge[1]]['opcode']:
                            predecessors_memory.append(edge[1])
            for edge in self.G.in_edges(node):
                if "virtual" not in self.G.edge[edge[0]][edge[1]]['opcode']:
                    oprandlist.append(edge[0])
                    opcode = self.G.edge[edge[0]][edge[1]]['opcode']
            if opcode == "icmp" or opcode == "fcmp":
                    predecessors_control.extend(oprandlist)
                    predecessors_control.append(node)
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

        #------------------------------------
        # to test the scalability of the ePVF
        #------------------------------------
        print "TEST"
        print len(predecessors_memory)
        print len(predecessors_control)
        #No need for selecting
        #print predecessors_memory
        #predecessors_memory = self.ordered_subset_bycycle(predecessors_memory,K)
        #predecessors_control = self.ordered_subset_bycycle(predecessors_control,K)
        #print predecessors_memory
        #print "TEST"
        #print len(predecessors_memory)
        #print len(predecessors_control)
        ReG = self.G.reverse()
        print len(self.G.nodes())
        p_set = set(predecessors_memory)
        predecessors_memory = list(p_set)
        print len(predecessors_memory)
        print len(ReG.nodes())
        sub_nodes = self.bfs(ReG,predecessors_memory)
        subG = self.G.subgraph(sub_nodes)
        print "Data flow graph"
        print len(subG)
        #processes = [Process(target=self.do_work, args=(ReG,target,sub_nodes)) for target in predecessors]
        #for p in processes:
        #    p.start()
        #for p in processes:
        #    p.join()
        p_set = set(predecessors_control)
        predecessors_control = list(p_set)
        #for target in predecessors_control:
        #    if target in ReG:
        #        T = nx.bfs_tree(ReG, target)
        #        tnodes = T.nodes()
        #        control_nodes.extend(tnodes)
        #        ReG.remove_nodes_from(tnodes)
        control_nodes = self.bfs(ReG,predecessors_control)
        subControlG = self.G.subgraph(control_nodes)
        print "Control flow graph"
        print len(subControlG)
        # source = []
        # for node in subG:
        #     if subG.in_degree(node) == 0:
        #         source.append(node)
        #     if subG.in_degree(node) == 1:
        #         for edge in subG.in_edges(node):
        #             if subG.edge[edge[0]][edge[1]]['opcode'] == 'virtual':
        #                 source.append(edge[1])
        predecessors.extend(predecessors_control)
        predecessors.extend(predecessors_memory)
        p_set = set(predecessors)
        predecessors = list(p_set)
        #for target in predecessors:
        #  if target in ReG:
        #     T = nx.bfs_tree(ReG, target)
        #     tnodes = T.nodes()
        #     sub_nodes.extend(tnodes)
        #     ReG.remove_nodes_from(tnodes)
             #sub_nodes.extend(T.nodes())
        uG_nodes = self.bfs(ReG,predecessors)
        nodes = set(uG_nodes)
        uG_nodes = list(nodes)
        uG = self.G.subgraph(uG_nodes)
        print "Union Graph"
        print len(uG)
        #self.simplePVF(subG, subControlG)
        #self.simplePVF(subControlG,subG)
        for node in uG.nodes_iter():
            numOut = 0
            for edge in uG.out_edges(node):
                if "virtual" not in uG.edge[edge[0]][edge[1]]['opcode']:
                    numOut += 1

            #if numOut == 0:
            #    print node
            #    for edge in uG.in_edges(node):
            #            if "alloca" == uG.edge[edge[0]][edge[1]]['opcode']:
            #                numOut += 1
            uG.node[node]['out_edge'] = numOut
        print time.time()
        print
        #for node in subG.nodes_iter():
        #    numOut = 0
        #    for edge in subG.out_edges(node):
        #        if "virtual" not in subG.edge[edge[0]][edge[1]]['opcode']:
        #            numOut += 1
        #    subG.node[node]['out_edge'] = numOut
        if lower_bound == 1:
            self.simplePVF(subG,subControlG)
        else:
            self.simplePVF(uG,subG)
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
            opcode = ""
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
            else:
                if opcode == "load":
                    for op in oplist:
                        for edge in G.in_edges(op):
                                opcode_new = G.edge[edge[0]][edge[1]]['opcode']
                                if opcode_new == "store":
                                    if G.node[edge[0]]['out_edge'] > 0 or len(G.in_edges(G.node[edge[0]])) == 0:
                                        if edge[0] not in rangeList:
                                            rangeList[edge[0]] = []
                                            rangeList[edge[0]].append(rangeList[node][0])
                                            rangeList[edge[0]].append(rangeList[node][1])
                                            finalBits.append(self.checkRange(G,edge[0],rangeList[node][0], rangeList[node][1], int(G.node[edge[0]]['len'])))
                                            G.node[edge[0]]['out_edge'] = int(G.node[edge[0]]['out_edge']) -1
                                            stack4recursion.append(edge[0])
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
        global ranking
        global control_start
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
        for item in oplist:
            if isfloat(G.node[item]['value']):
                G.node[item]['value'] = int(float(G.node[item]['value']))
        max_range = int(rangeList[node][0])
        min_range = int(rangeList[node][1])
        if opcode == "add" or opcode == "fadd":
            assert(len(oplist) == 2)
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            for i in range(2):
                    max_op = max_range - int(G.node[oplist[1-i]]['value'])
                    min_op = min_range - int(G.node[oplist[1-i]]['value'])
                    if oplist[i] not in rangeList:
                        rangeList[oplist[i]] = []
                        rangeList[oplist[i]].append(max_op)
                        rangeList[oplist[i]].append(min_op)
                    type = G.node[oplist[i]]['len']
                    ace_inst += type
                    #if "constant" not in oplist[i]:
                    if int(G.node[oplist[i]]['out_edge']) > 0:
                        G.node[oplist[i]]['out_edge'] = int(G.node[oplist[i]]['out_edge']) -1
                        crash_tmp = self.checkRange(G,oplist[i],max_op, min_op, type)
                        finalBits.append(crash_tmp)
                        crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])


        if opcode == "sub" or opcode == "fsub":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            assert(len(oplist) == 2)
            # the first operand
            max_op = max_range + int(G.node[oplist[1]]['value'])
            min_op = min_range + int(G.node[oplist[1]]['value'])
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            #if "constant" not in oplist[0]:
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0],max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            max_op = int(G.node[oplist[0]]['value']) - min_range
            min_op = int(G.node[oplist[0]]['value']) - max_range
            if oplist[1] not in rangeList:
                rangeList[oplist[1]] = []
                rangeList[oplist[1]].append(max_op)
                rangeList[oplist[1]].append(min_op)
            type = G.node[oplist[1]]['len']
            ace_inst += type
            #if "constant" not in oplist[1]:
            if int(G.node[oplist[1]]['out_edge']) > 0:
                G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[1],max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "fmul" or opcode == "mul":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
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
                    ace_inst += type
                    #if "constant" not in oplist[i]:
                    if int(G.node[oplist[i]]['out_edge']) > 0:
                        G.node[oplist[i]]['out_edge'] = int(G.node[oplist[i]]['out_edge']) -1
                        crash_tmp = self.checkRange(G,oplist[i],max_op, min_op, type)
                        finalBits.append(crash_tmp)
                        crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "udiv"  or opcode == "sdiv" or opcode == "fdiv":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            assert(len(oplist) == 2)
            # the first operand
            max_op = max_range*int(G.node[oplist[1]]['value'])
            min_op = min_range*int(G.node[oplist[1]]['value'])
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            #if "constant" not in oplist[0]:
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0] ,max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            max_op = int(G.node[oplist[0]]['value'])/min_range
            min_op = int(G.node[oplist[0]]['value'])/max_range
            if oplist[1] not in rangeList:
                rangeList[oplist[1]] = []
                rangeList[oplist[1]].append(max_op)
                rangeList[oplist[1]].append(min_op)
            type = G.node[oplist[1]]['len']
            ace_inst += type
            #if "constant" not in oplist[1]:
            if int(G.node[oplist[1]]['out_edge']) > 0:
                G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[1] ,max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "sext":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0],max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "phi":
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            #finalBits.append(self.checkRange(G, oplist[0] ,max_op, min_op, type))

        if opcode == "trunc":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0] ,max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "srem":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
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
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                finalBits.append(self.checkRange(G, oplist[0] ,max(tmp), min(tmp), type))
            type = int(G.node[oplist[1]]['len'])
            ace_inst += type
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
            if int(G.node[oplist[1]]['out_edge']) > 0:
                G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[1] ,max(tmp), min(tmp), type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "getelementptr":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
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
                    ace_inst += type
                    #if "constant" not in oplist[0]:
                    if int(G.node[oplist[0]]['out_edge']) > 0:
                        G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                        crash_tmp = self.checkRange(G, oplist[0],max_op, min_op, type)
                        finalBits.append(crash_tmp)
                        crash_inst += crash_tmp
                    max_op = (max_range - int(G.node[oplist[0]]['value']))*8/int(size)
                    min_op = (min_range - int(G.node[oplist[0]]['value']))*8/int(size)
                    if oplist[1] not in rangeList:
                        rangeList[oplist[1]] = []
                        rangeList[oplist[1]].append(max_op)
                        rangeList[oplist[1]].append(min_op)
                    type = G.node[oplist[1]]['len']
                    ace_inst += type
                    #if "constant" not in oplist[1]:
                    if int(G.node[oplist[1]]['out_edge']) > 0:
                        G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                        crash_tmp = self.checkRange(G, oplist[1] ,max_op, min_op, type)
                        finalBits.append(crash_tmp)
                        crash_inst += crash_tmp
                    if _index in ranking:
                        ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
                    else:
                        ranking[_index] = []
                        ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

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
                        ace_inst += type
                        #if "constant" not in oplist[0]:
                        if int(G.node[oplist[0]]['out_edge']) > 0:
                            G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                            crash_tmp = self.checkRange(G, oplist[0] ,max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(t/8))*8/int(size)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(t/8))*8/int(size)
                        if oplist[1] not in rangeList:
                            rangeList[oplist[1]] = []
                            rangeList[oplist[1]].append(max_op)
                            rangeList[oplist[1]].append(min_op)
                        type = G.node[oplist[1]]['len']
                        ace_inst += type
                        #if "constant" not in oplist[1]:
                        if int(G.node[oplist[1]]['out_edge']) > 0:
                            G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                            crash_tmp = self.checkRange(G, oplist[1],max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/8
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/8
                        if oplist[2] not in rangeList:
                            rangeList[oplist[2]] = []
                            rangeList[oplist[2]].append(max_op)
                            rangeList[oplist[2]].append(min_op)
                        type = G.node[oplist[2]]['len']
                        ace_inst += type
                        #if "constant" not in oplist[2]:
                        if int(G.node[oplist[2]]['out_edge']) > 0:
                            G.node[oplist[2]]['out_edge'] = int(G.node[oplist[2]]['out_edge']) -1
                            crash_tmp = self.checkRange(G, oplist[2],max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        if _index in ranking:
                            ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
                        else:
                            ranking[_index] = []
                            ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

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
                        ace_inst += type
                        #if "constant" not in oplist[0]:
                        if int(G.node[oplist[0]]['out_edge']) > 0:
                            G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                            crash_tmp = self.checkRange(G, oplist[0],max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - t)/int(size)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - t)/int(size)
                        if oplist[1] not in rangeList:
                            rangeList[oplist[1]] = []
                            rangeList[oplist[1]].append(max_op)
                            rangeList[oplist[1]].append(min_op)
                        type = G.node[oplist[1]]['len']
                        ace_inst += type
                        #if "constant" not in oplist[1]:
                        if int(G.node[oplist[1]]['out_edge']) > 0:
                            G.node[oplist[1]]['out_edge'] = int(G.node[oplist[1]]['out_edge']) -1
                            crash_tmp = self.checkRange(G,oplist[1],max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        max_op = (max_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/int(int(element)/8)
                        min_op = (min_range - int(G.node[oplist[0]]['value']) - int(G.node[oplist[1]]['value'])*int(size))/int(int(element)/8)
                        if oplist[2] not in rangeList:
                            rangeList[oplist[2]] = []
                            rangeList[oplist[2]].append(max_op)
                            rangeList[oplist[2]].append(min_op)
                        type = G.node[oplist[2]]['len']
                        ace_inst += type
                        #if "constant" not in oplist[2]:
                        if int(G.node[oplist[2]]['out_edge']) > 0:
                            G.node[oplist[2]]['out_edge'] = int(G.node[oplist[2]]['out_edge']) -1
                            crash_tmp = self.checkRange(G, oplist[2],max_op, min_op, type)
                            finalBits.append(crash_tmp)
                            crash_inst += crash_tmp
                        if _index in ranking:
                            ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
                        else:
                            ranking[_index] = []
                            ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])



        if opcode == "bitcast":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0], max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "alloca":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0], max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "call":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0], max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "fptosi":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0], max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])

        if opcode == "sitofp":
            cycle = int(G.node[node]['cycle'])
            _index = self.cycle_index_lookup[cycle]
            crash_inst = 0
            ace_inst = 0
            max_op = max_range
            min_op = min_range
            if oplist[0] not in rangeList:
                rangeList[oplist[0]] = []
                rangeList[oplist[0]].append(max_op)
                rangeList[oplist[0]].append(min_op)
            type = G.node[oplist[0]]['len']
            ace_inst += type
            if int(G.node[oplist[0]]['out_edge']) > 0:
                G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                crash_tmp = self.checkRange(G, oplist[0], max_op, min_op, type)
                finalBits.append(crash_tmp)
                crash_inst += crash_tmp
            if _index in ranking:
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])
            else:
                ranking[_index] = []
                ranking[_index].append([cycle,ace_inst-crash_inst,ace_inst])


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
        temp_s = G.node[node]['value']
        if isfloat(temp_s):
            address = int(float(temp_s))+1
        else:
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
        bitlist = []
        for i in range(type):
            mask = (1 << i)
            new = address
            new = new ^ mask
            if new > max_f:
                counter += 1
                bitlist.append(i)
            if new < min_f:
                if esp == 0:
                    counter += 1
                    bitlist.append(i)
                    min_new = min_f
                else:
                    if new < esp - 65536 - 128:
                        counter += 1
                        bitlist.append(i)
                        min_new = min(esp - 65536 - 128, min_f)
        return [counter, min_new, bitlist]

    def instructionPVF(self, G, refG, opcode, oplist, node):
        global crashBits
        global stack
        global loadstore
        global rangeList
        global finalBits
        global loadstore_bits
        bb = 0
        removed = 0
        removed_ldst = 0
        if opcode in config.computationInst:
            for op in oplist:
                #if "constant" in op:
                #    continue
                res = int(G.node[op]['len'])
                if opcode in config.floatingPoint:
                    res = int(res)*f_level
                bb += int(res)
            if opcode == "mul" or opcode == "fmul":
                    for index, op in enumerate(oplist):
                        if iszero(G.node[op]['value']) :
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
                shift = int(G.node[oplist[1]]['value'])
                bb += math.log(size,2)
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
                        bb += type
                        max = 0
                        min = 0
                        removed1 = 0
                        start_node = ""
                        #for item1, item2, item3 in grouper(3, range_mem):
                        #print range_mem
                        #print address
                        #print "+++++++"
                        bitlist = []
                        for key in range_mem:
                            if key == 'heap':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min, bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 'stack':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                esp = range_mem['esp'][0]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),int(esp), type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 0:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 1:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                        if max == 0 or min == 0:
                            print "wrong!!!!"
                        #if iszero(G.node[node]['value']):
                        #    max = 0
                        #    min = 0
                        #    removed1 = 64
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
                        #for edge in G.in_edges(op):
                        #        # for alloca
                        #    if "root" in edge[0]:
                        #        self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                        #        break
                        #    if "mem" in G.node[edge[0]]:
                        #        self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                        #        #removed += removed1
                        #        break
                        #    if edge[0] in config.Outbound:
                        #        finalBits.append(self.checkRange(G,start_node,max,min,int(G.node[start_node]['len'])))
                                #removed += removed1
                        #        break
                        #if len(G.in_edges(op)) == 0:
                        #    self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                        #removed += removed1
                        #if int(G.node[op]['out_edge']) > 0:
                        loadstore.append(removed1)
                        removed_ldst = removed1
                        G.node[op]['out_edge'] = int(G.node[op]['out_edge']) -1
                        loadstore_bits[op] = bitlist
                        chain = 0
                        crash = 0
                        for i in finalBits:
                            chain += i
                        for i in loadstore:
                            crash += i
                        #print chain
                        #print crash
                        #print node
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
                    bitlist = []
                    for key in range_mem:
                            if key == 'heap':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 0:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 1:
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),0, type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                            if key == 'stack':
                                item1 = range_mem[key][0]
                                item2 = range_mem[key][1]
                                esp = range_mem['esp'][0]
                                if int(item1) <= int(address) and int(address) <= int(item2):
                                    [removed1,new_min,bitlist] = self.checkRange1(int(address), int(item2), int(item1),int(esp), type)
                                    max = int(item2)
                                    min = int(new_min)
                                    break
                    for edge in G.in_edges(node):
                            if G.edge[edge[0]][edge[1]]['opcode'] == "store":
                                if "post" in G.node[edge[0]]:
                                    op = G.node[edge[0]]['post']
                                    #if G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] in config.pointerInst or G.edge[edge_nxt[0]][edge_nxt[1]]['opcode'] == "alloca":
                                    #     if flag == 0:
                                    #       removed += removed
                                    #        flag = 1
                                    #        break
                                    bb += int(G.node[edge[0]]['len'])
                                    self.getParent4CrashChain(G, op,max, min)
                                    #print node
                                    #print op
                                    #print "++++++"
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
                    #for edge in G.in_edges(node):
                    #            # for alloca
                    #        if "root" in edge[0]:
                    #            self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    #            break
                    #        if "mem" in G.node[edge[0]]:
                    #            self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    #            #removed += removed1
                    #            break
                    #        if edge[0] in config.Outbound:
                    #            self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                                #removed += removed1
                    #            break
                    #if len(G.in_edges(node)) == 0:
                    #    self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    #removed += removed1
                    #if int(G.node[node]['out_edge']) > 0:
                    loadstore.append(removed1)
                    removed_ldst = removed1
                    G.node[oplist[0]]['out_edge'] = int(G.node[oplist[0]]['out_edge']) -1
                    loadstore_bits[node] = bitlist
                    chain = 0
                    crash = 0
                    for i in finalBits:
                            chain += i
                    for i in loadstore:
                            crash += i

                    #print chain
                    #print crash
                    #print node
                    #self.checkRange(G,start_node,max,min,int(G.node[start_node]['len']))
                    stack = []

        elif opcode in config.castInst:
            if opcode == "fptrunc" or opcode == "trunc":
                res = G.node[node]['len']
                bb += res
            else:
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
                ## need to consider the real result of the comparison, shouldn't matter too much
                bool_res = int(G.node[node]['value'])
                bb += self.checkCMPRange(G, refG, oplist, predicate, bool_res)
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
        _cycle = int(G.node[node]['cycle'])
        _index = self.cycle_index_lookup[int(G.node[node]['cycle'])]
        ace_inst = 0
        for op in oplist:
            ace_inst += int(G.node[op]['len'])
        if _index in ranking:
            ranking[_index].append([_cycle,bb,ace_inst])
        else:
            ranking[_index] = []
            ranking[_index].append([_cycle,bb,ace_inst])
        #if opcode not in config.memoryInst:
        #    bb += G.node[node]['len']
        #print b
        #print "######"
        #print node
        #print oplist
        #print counter
        return bb

    def checkCMPRange_LB(self,G,refG,oplist,predicate):
        global finalBits_control
        oplist.reverse()

        oplist_new = []
        opcode = ""
        stack4recursion = []
        stack4recursion.extend(oplist)
        final = 0
        for op in oplist:
            final += int(G.node[op]['len'])
            G.node[op]['out_edge'] = int(G.node[op]['out_edge']) -1
        finalBits_control.append(final)
        while len(stack4recursion) != 0:
            node = stack4recursion.pop()
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oplist_new.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            if opcode != "load" and opcode != "store":
                for op in oplist_new:
                        #self.getParent4CrashChain(G, op, cbits)

                        stack4recursion.append(op)
                        if G.node[op]['out_edge'] > 0:
                                size = G.node[op]['len']
                                finalBits_control.append(size)
                                G.node[op]['out_edge'] = G.node[op]['out_edge'] -1
            else:
                if opcode == "load":
                    for op in oplist_new:
                        for edge in G.in_edges(op):
                                opcode_new = G.edge[edge[0]][edge[1]]['opcode']
                                if opcode_new == "store":
                                    stack4recursion.append(edge[0])
                                    if G.node[edge[0]]['out_edge'] > 0:
                                            finalBits_control.append(int(G.node[edge[0]]['len']))
                                            G.node[edge[0]]['out_edge'] = G.node[edge[0]]['out_edge'] -1
            oplist_new = []
        return final



    def checkCMPRange(self,G,refG,oplist,predicate, bool_res):
        count_1 = 0
        count_2 = 0
        global finalBits_control
        oplist.reverse()
        if int(predicate) == 40:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new < value2:
                        count_1 += 1
                else:
                    if new >= value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 < new:
                        count_2 += 1
                else:
                    if value1 >= new:
                        count_2 += 1
        if int(predicate) == 41:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new <= value2:
                        count_1 += 1
                else:
                    if new > value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 <= new:
                        count_2 += 1
                else:
                    if value1 > new:
                        count_2 += 1
        if int(predicate) == 39:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new >= value2:
                        count_1 += 1
                else:
                    if new < value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 >= new:
                        count_2 += 1
                else:
                    if value1 < new:
                        count_2 += 1
        if int(predicate) == 32:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new == value2:
                        count_1 += 1
                else:
                    if new != value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 == new:
                        count_2 += 1
                else:
                    if value1 != new:
                        count_2 += 1
        if int(predicate) == 33:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new != value2:
                        count_1 += 1
                else:
                    if new == value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 != new:
                        count_2 += 1
                else:
                    if value1 == new:
                        count_2 += 1
        if int(predicate) == 38:
            value1 = int(G.node[oplist[0]]['value'])
            value2 = int(G.node[oplist[1]]['value'])
            for i in range(int(G.node[oplist[0]]['len'])):
                mask = (1 << i)
                new = value1
                new = new ^ mask
                if bool_res >= 0:
                    if new >= value2:
                        count_1 += 1
                else:
                    if new < value2:
                        count_1 += 1
            #if "constant" not in oplist[1]:
            for i in range(G.node[oplist[1]]['len']):
                mask = (1 << i)
                new = value2
                new = new ^ mask
                if bool_res >= 0:
                    if value1 >= new:
                        count_2 += 1
                else:
                    if value1 < new:
                        count_2 += 1

        oplist_new = []
        opcode = ""
        stack4recursion = []
        stack4recursion.extend(oplist)
        final = 0
        for op in oplist:
            final += int(G.node[op]['len'])
        icmpbits = {}
        icmpbits[count_1] = []
        icmpbits[count_1].append(oplist[0])
        icmpbits[count_2] = []
        icmpbits[count_2].append(oplist[1])
        finalBits_control.append(count_1)
        finalBits_control.append(count_2)
        while len(stack4recursion) != 0:
            node = stack4recursion.pop()
            for edge in G.in_edges(node):
                if "virtual" not in G.edge[edge[0]][edge[1]]['opcode']:
                    oplist_new.append(edge[0])
                    opcode = G.edge[edge[0]][edge[1]]['opcode']
            if opcode != "load" and opcode != "store":
                for op in oplist_new:
                        #self.getParent4CrashChain(G, op, cbits)
                        if op not in refG:
                            for key in icmpbits:
                                if G.node[op]['out_edge'] > 0:
                                    if node in icmpbits[key]:
                                        if opcode == "trunc":
                                            size = G.node[op]['len'] - G.node[node]['len']+key
                                            final -= key*(len(icmpbits[key]))
                                            icmpbits.pop(key)
                                            if size not in icmpbits:
                                                icmpbits[size] = []
                                                icmpbits[size].append(op)
                                                finalBits_control.append(size)
                                            else:
                                                icmpbits[size].append(op)
                                                finalBits_control.append(size)

                                        elif opcode == "sext":
                                            size = G.node[node]['len'] - G.node[op]['len']
                                            if key >= size:
                                                final -= key*(len(icmpbits[key]))
                                                icmpbits.pop(key)
                                                if size not in icmpbits:
                                                    icmpbits[size] = []
                                                    icmpbits[size].append(op)
                                                    finalBits_control.append(size)
                                                else:
                                                    icmpbits[size].append(op)
                                                    finalBits_control.append(size)
                                            else:
                                                icmpbits[key].append(op)
                                                finalBits_control.append(key)

                                        else:
                                            icmpbits[key].append(op)
                                            finalBits_control.append(key)
                                    G.node[op]['out_edge'] = G.node[op]['out_edge'] -1
                                    stack4recursion.append(op)
                                    break
                        else:
                            if G.node[op]['out_edge'] > 0:
                                for key in icmpbits:
                                    if node in icmpbits[key]:
                                        icmpbits[key].append(op)
                                        finalBits_control.append(key)
                                        stack4recursion.append(op)
                                        G.node[op]['out_edge'] = G.node[op]['out_edge'] -1
            else:
                if opcode == "load":
                    for op in oplist_new:
                        for edge in G.in_edges(op):
                                opcode_new = G.edge[edge[0]][edge[1]]['opcode']
                                if opcode_new == "store":

                                    if edge[0] not in refG:
                                        if G.node[edge[0]]['out_edge'] > 0:
                                            for key in icmpbits:
                                                if node in icmpbits[key]:
                                                    icmpbits[key].append(edge[0])
                                                    finalBits_control.append(key)
                                                    stack4recursion.append(edge[0])
                                                    G.node[edge[0]]['out_edge'] = G.node[edge[0]]['out_edge'] -1
                                    else:
                                        if G.node[edge[0]]['out_edge'] > 0:
                                            for key in icmpbits:
                                                 if node in icmpbits[key]:
                                                     icmpbits[key].append(edge[0])
                                                     finalBits_control.append(key)
                                                     G.node[edge[0]]['out_edge'] = G.node[edge[0]]['out_edge'] -1
                                                     stack4recursion.append(edge[0])
            oplist_new = []
        return final

    def simplePVF(self,G,refG):
        global rangeList
        global finalBits
        global loadstore
        global finalBits_control
        global control_start
        global pop
        b = 0
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
                a = self.instructionPVF(G, refG, opcode, oprandlist, node)
                #print "+++++"
                #print opcode
                #print node
                #print oprandlist
                #print a
                b += a
                count += 1
        crash = 0
        ldst = 0
        control = 0
        print count
        if lower_bound == 1:
            for node in refG.nodes_iter():
                oprandlist = []
                opcode = ""
                for edge in refG.in_edges(node):
                    if "virtual" not in refG.edge[edge[0]][edge[1]]['opcode']:
                        oprandlist.append(edge[0])
                        opcode = refG.edge[edge[0]][edge[1]]['opcode']
                if opcode == "load" or opcode == "store":
                #print "======"
                #print opcode
                #print oprandlist
                    self.instructionPVF(refG, refG, opcode, oprandlist, node)
                    count += 1
        print count
        for i in finalBits_control:
            control += i
        for i in finalBits:
            crash += i
        for i in loadstore:
            ldst += i
        #for node in G.nodes_iter():
        #    if "out_edge" in G.node[node]:
        #        if G.node[node]['out_edge'] != 0:
        #            print node
        #            print G.node[node]['out_edge']
        print "Control chain"
        print control
        print "Crash chain"
        print crash
        print "Crash on ld st"
        print ldst
        print "SDC bits"
        print b
        print (b-ldst-crash-control)
        # generating duplication candidates
        f_full = open("full_duplication","w")
        f_random = open("random_duplication","w")
        f_hotpath = open("hotpath_duplication","w")
        f_epvf = open("epvf_duplication","w")
        for item in ranking:
            f_full.write(str(item)+"\n")
        f_full.close()
        random_list = random.sample(ranking.keys(),int(pop*len(ranking)))
        for item in random_list:
            f_random.write(str(item)+"w")
        f_random.close()
        hotpath_index = OrderedDict(sorted(ranking.viewitems(), key=lambda x: len(x[1]), reverse=True))
        count = 0
        for item in hotpath_index:
            f_hotpath.write(str(item)+"\n")
            count += 1
            if count > int(pop*len(ranking)):
                break
        f_hotpath.close()
        epvf_dict = {}
        count = 0
        for key, value in ranking:
            ace = 0
            total = 0
            for i in value:
                ace += i[1]
                total += i[2]
            epvf = float(ace)/total
            epvf_dict[key] = epvf
        ordered_epvf_dict = OrderedDict(sorted(epvf_dict.viewitems(), key=lambda x: x[1]),reverse = True)
        for item in ordered_epvf_dict:
            f_epvf.write(str(item)+"\n")
            count += 1
            if count > len(ranking):
                break
        f_epvf.close()
        #self.crashRecall(G,config.crashfile)


    def crashRecall(self,G,cfile):
        global rangeList
        global loadstore_bits
        count_crash = 0
        count_pre = 0
        #print "LOL"
        #print self.global_hash_cycle
        with open(cfile,"r") as cf:
            lines = cf.readlines()
            for line in lines:
                if "fi_cycle" in line:
                    index = re.findall('fi_index=\d+',line)[0].split("=")[1]
                    cycle = re.findall('fi_cycle=\d+',line)[0].split("=")[1]
                    reg_index = re.findall('fi_reg_index=\d+',line)[0].split("=")[1]
                    bit = int(re.findall('fi_bit=\d+',line)[0].split("=")[1])
                    flag = 0
                    flag1 = 0
                    if int(cycle)+1 not in self.global_hash_cycle:
                        continue
                    if "store" in self.global_hash_cycle[int(cycle)+1][len(self.global_hash_cycle[int(cycle)+1])-1] and "constant" in self.global_hash_cycle[int(cycle)+1][0]:
                        reg_index = '1'
                    for node in self.global_hash_cycle[int(cycle)+1]:
                        if node in G:
                                if 'index' in G.node[node]:
                                    if int(reg_index) == int(G.node[node]['index']):

                                        if node in rangeList:
                                            if len(rangeList[node]) > 2:
                                                count_crash += 1
                                                if bit in rangeList[node][2]:
                                                    count_pre += 1
                                                else:
                                                    print "wrong bit in chain"
                                        elif node in loadstore_bits:
                                            count_crash += 1
                                            if bit in loadstore_bits[node]:
                                                count_pre += 1
                                            else:
                                                print "wrong bit in ldst"
                                        else:
                                            count_crash+=1
                                            print "control flow effect"
                                            print cycle
                                            print node
                                    else:
                                        flag += 1

                                if "stindex" in G.node[node] and  "store" in self.global_hash_cycle[int(cycle)+1][len(self.global_hash_cycle[int(cycle)+1])-1]:
                                        if int(reg_index) == int(G.node[node]['stindex']):
                                            if node in rangeList:
                                                if len(rangeList[node]) > 2:
                                                    count_crash += 1
                                                    if bit in rangeList[node][2]:
                                                        count_pre += 1
                                                    else:
                                                        print "wrong bit in chain"
                                            elif node in loadstore_bits:
                                                count_crash += 1
                                                if bit in loadstore_bits[node]:
                                                    count_pre += 1
                                                else:
                                                    print "wrong bit in ldst"
                                            else:
                                                 count_crash+=1
                                                 print "here1"
                                            flag -= 1
                        else:
                                flag1 += 1
                    if flag1 == len(self.global_hash_cycle[int(cycle)+1]):
                        print "CYCLE"
                        print cycle
                        print "Wrong"
        print count_pre
        print count_crash
        #generate data for percision
        len_chain = len(rangeList)
        len_crash = len(loadstore_bits)
        total = 10000
        r_chain = int(10000*(float(len_chain)/float((len_chain+len_crash))))
        r_crash = int(10000*(float(len_crash)/float((len_chain+len_crash))))
        f = open(config.precision_file,"w")
        output = []
        while r_chain > 0:
                key = random.sample(self.global_hash_cycle,1)[0]
                for node in self.global_hash_cycle[key]:
                    if node in rangeList:
                            if "constant" in node:
                                continue
                            if len(rangeList[node]) < 3:
                                continue
                            cycle = key
                            index = 0
                            if "stindex" in G.node[node] and "store" in self.global_hash_cycle[int(cycle)][len(self.global_hash_cycle[int(cycle)])-1]:
                                if "constant" in self.global_hash_cycle[int(cycle)][0]:
                                    index = 0
                                else:
                                    index = int(G.node[node]['stindex'])
                            else:
                                index = G.node[node]['index']
                            bit = random.sample(rangeList[node][2],1)[0]
                            output.append("fi_cycle="+str(cycle)+" fi_reg_index="+str(index)+" fi_bit="+str(bit)+"\n")
                            r_chain -= 1
        while r_crash > 0:
                key = random.sample(self.global_hash_cycle,1)[0]
                for node in self.global_hash_cycle[key]:
                    if node in loadstore_bits:
                            if "constant" in node:
                                continue
                            cycle = key
                            index = 0
                            if "stindex" in G.node[node] and "store" in self.global_hash_cycle[int(cycle)][len(self.global_hash_cycle[int(cycle)])-1]:
                                if "constant" in self.global_hash_cycle[int(cycle)][0]:
                                    index = 0
                                else:
                                    index = int(G.node[node]['stindex'])
                            else:
                                index = G.node[node]['index']
                            bit = random.sample(loadstore_bits[node],1)[0]
                            output.append("fi_cycle="+str(cycle)+" fi_reg_index="+str(index)+" fi_bit="+str(bit)+"\n")
                            r_crash -= 1
        f.writelines(output)
        f.close()