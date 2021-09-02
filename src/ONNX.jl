module ONNX
  const _ProtoBuf_Top_ = @static isdefined(parentmodule(@__MODULE__), :_ProtoBuf_Top_) ? (parentmodule(@__MODULE__))._ProtoBuf_Top_ : parentmodule(@__MODULE__)
  include("onnx_pb.jl")
  include("read.jl")
  include("write.jl")
  include("conversions.jl")
  include("show.jl")
  include("ops.jl")
  include("load.jl")
  include("save.jl")
end
