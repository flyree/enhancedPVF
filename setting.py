
#### pathfinder
#IRpath = "/Users/bofang/PycharmProjects/PVF/pathfinder/pathfinder.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/pathfinder/pathfinder_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/pathfinder/llfi.stat.trace.txt.prof.1"
#outputDataSet = ['.omp_microtask._%arrayidx46']
#tid = "18608"
#Outbound = ['.omp_microtask._@wall','.omp_microtask._@cols']

### bfs
IRpath = "/Users/bofang/PycharmProjects/PVF/bfs/bfs.ll"
indexFilePath = "/Users/bofang/PycharmProjects/PVF/bfs/bfs_mapping"
tracePath = "/Users/bofang/PycharmProjects/PVF/bfs/llfi.stat.trace.txt.prof.1"
outputDataSet = ['.omp_microtask._%arrayidx40']
Outbound = ['.omp_microtask._%2','.omp_microtask._@no_of_nodes']
tid = "23923"

### hotspot
#IRpath = "/Users/bofang/PycharmProjects/PVF/hotspot/hotspot.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/hotspot/hotspot_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/hotspot/llfi.stat.trace.txt.prof.1"
#outputDataSet = ['.omp_microtask.1_%arrayidx22']
#tid = "5142"

### reduction
#IRpath = "/Users/bofang/PycharmProjects/PVF/reduction/reduction.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/reduction/reduction_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/reduction/llfi.stat.trace.txt.prof.1"
#outputDataSet = ['.omp_microtask._%arrayidx']
#tid = "8819"

#-------------
# instructions
#-------------
OSbits = 64
computationInst = ['add','fadd','sub','fsub','mul', 'fmul','udiv', 'sdiv', 'fdiv', 'urem', 'srem', 'frem']
bitwiseInst = ['shl', 'lshr', 'ashr', 'and', 'or', 'xor']
pointerInst = ['getelementptr']
memoryInst = ['load', 'store', 'alloca']
castInst = ['zext', 'sext', 'fptrunc', 'fpext', 'fptoui', 'fptosi', 'uitofp', 'sitofp', 'ptrtoint', 'inttoptr', 'bitcast', 'addrspacecast']
otherInst = ['icmp', 'fcmp', 'phi', 'select', 'call']


intrinsics = ['memcpy', 'atoi', 'printf', 'exit', 'llvm.umul.with.overflow.i64', '_Znam', 'srand',
              '__kmpc_dispatch_fini_4', '__kmpc_barrier', '__kmpc_dispatch_next_4', 'llvm.memcpy.p0i8.p0i8.i64',
              '_ZdaPv', 'gettimeofday', '__kmpc_fork_call', 'ompc_set_num_threads', ]