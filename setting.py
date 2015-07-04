
#### pathfinder
#IRpath = "/Users/bofang/PycharmProjects/PVF/pathfinder/pathfinder.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/pathfinder/pathfinder_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/pathfinder/llfi.stat.trace.txt.prof.0"
#outputDataSet = ['.omp_microtask._%arrayidx46']
#crashfile = "/Users/bofang/PycharmProjects/PVF/pathfinder/crash"
#precision_file = "/Users/bofang/PycharmProjects/PVF/pathfinder/precision"
#tid = "18608"
#Outbound = ['.omp_microtask._@wall','.omp_microtask._@cols']
#gepsize = 4
### lavaMD
#IRpath = "/Users/bofang/PycharmProjects/PVF/lavaMD/lavaMD.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/lavaMD/lavaMD_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/lavaMD/llfi.stat.trace.txt.prof.0"
#outputDataSet = ['.omp_microtask._%v114','.omp_microtask._%x121','.omp_microtask._%y128','.omp_microtask._%z135']
#crashfile = "/Users/bofang/PycharmProjects/PVF/lavaMD/crash"
#precision_file = "/Users/bofang/PycharmProjects/PVF/lavaMD/precision"
#tid = "18608"
#Outbound = ['.omp_microtask._@wall','.omp_microtask._@cols']

### particle
#IRpath = "/Users/bofang/PycharmProjects/PVF/particle/particle_filter.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/particle/mapping_particle"
#tracePath = "/Users/bofang/PycharmProjects/PVF/particle/llfi.stat.trace.txt.prof.1"
#outputDataSet = ['.omp_microtask._%arrayidx110']
#Outbound = ['.omp_microtask._%2','.omp_microtask._%0']
#crashfile = "/Users/bofang/PycharmProjects/PVF/particle/crash"
#precision_file = "/Users/bofang/PycharmProjects/PVF/particle/precision"

### sc
#IRpath = "/Users/bofang/PycharmProjects/PVF/sc/sc_omp.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/sc/mapping_sc"
#tracePath = "/Users/bofang/PycharmProjects/PVF/sc/llfi.stat.trace.txt.prof.0"
#outputDataSet = ['.omp_microtask._%assign49']
#Outbound = ['.omp_microtask._%2','.omp_microtask._%0']
#crashfile = "/Users/bofang/PycharmProjects/PVF/sc/crash"
#recision_file = "/Users/bofang/PycharmProjects/PVF/sc/precision"

### nw
IRpath = "/Users/bofang/PycharmProjects/PVF/nw/needle.ll"
indexFilePath = "/Users/bofang/PycharmProjects/PVF/nw/needle_mapping"
tracePath = "/Users/bofang/PycharmProjects/PVF/nw/llfi.stat.trace.txt.prof.0"
outputDataSet = ['.omp_microtask.9_%arrayidx42','.omp_microtask._%arrayidx40']
crashfile = "/Users/bofang/PycharmProjects/PVF/nw/crash"
precision_file = "/Users/bofang/PycharmProjects/PVF/nw/precision"
tid = "18608"
Outbound = ['.omp_microtask._@wall','.omp_microtask._@cols']
### bfs
#IRpath = "/Users/bofang/PycharmProjects/PVF/bfs/bfs.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/bfs/bfs_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/bfs/llfi.stat.trace.txt.prof.0.test"
#outputDataSet = ['.omp_microtask._%arrayidx40']
#Outbound = ['.omp_microtask._%2','.omp_microtask._@no_of_nodes']
#tid = "23923"

### hotspot
#IRpath = "/Users/bofang/PycharmProjects/PVF/hotspot/hotspot.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/hotspot/hotspot_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/hotspot/llfi.stat.trace.txt.prof.0"
#outputDataSet = ['.omp_microtask.1_%arrayidx22']
#Outbound = ['.omp_microtask._%0','.omp_microtask._%2', '.omp_microtask.1_%0','.omp_microtask.1_%2']
#precision_file = "/Users/bofang/PycharmProjects/PVF/hotspot/precision"
#tid = "2909"
#gepsize = 8

### matrix multiplication
#IRpath = "/Users/bofang/PycharmProjects/PVF/mm/mm.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/mm/mm_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/mm/llfi.stat.trace.txt.prof.0"
#outputDataSet = ['.omp_microtask._%arrayidx21']
#Outbound = ['.omp_microtask._%2','.omp_microtask._%0']
#crashfile = "/Users/bofang/PycharmProjects/PVF/mm/crash"
#precision_file = "/Users/bofang/PycharmProjects/PVF/mm/precision"

### reduction
#IRpath = "/Users/bofang/PycharmProjects/PVF/reduction/reduction.ll"
#indexFilePath = "/Users/bofang/PycharmProjects/PVF/reduction/reduction_mapping"
#tracePath = "/Users/bofang/PycharmProjects/PVF/reduction/llfi.stat.trace.txt.prof.1"
#outputDataSet = ['.omp_microtask._%arrayidx']
#Outbound = ['.omp_microtask._%2']
#tid = "8819"

#-------------
# instructions
#-------------
OSbits = 64
computationInst = ['add','fadd','sub','fsub','mul', 'fmul','udiv', 'sdiv', 'fdiv', 'urem', 'srem', 'frem','call']
floatingPoint = ['fadd','fmul','fdiv','fsub','frem']
bitwiseInst = ['shl', 'lshr', 'ashr', 'and', 'or', 'xor']
pointerInst = ['getelementptr']
memoryInst = ['load', 'store','alloca']
castInst = ['zext', 'sext', 'fptrunc', 'fpext', 'fptoui', 'fptosi', 'uitofp', 'sitofp', 'ptrtoint', 'inttoptr', 'bitcast', 'addrspacecast','trunc']
otherInst = ['icmp', 'fcmp', 'phi', 'select']


intrinsics = ['memcpy', 'printf', 'exit', 'llvm.umul.with.overflow.i64', '_Znam',
              '__kmpc_dispatch_fini_4', '__kmpc_barrier', '__kmpc_dispatch_next_4', 'llvm.memcpy.p0i8.p0i8.i64',
              '_ZdaPv', 'gettimeofday', '__kmpc_fork_call', 'ompc_set_num_threads']
extra = ['srand','atoi','itoa','roundDouble','llvm.pow.f64','fabs']