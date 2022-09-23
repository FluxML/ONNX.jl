import Umlaut: Tape, Input, Call, mkcall, V


"""
    rename_keys(dict::Dict, subs::Dict)

Create a copy of `dict`, replacing its keys according to mapping `subs`
"""
rename_keys(dct::Dict, subs::Dict) = Dict(get(subs, k, k) => v for (k, v) in pairs(dct))


"""
    unpacked_vars(tape::Umlaut.Tape, op::Umlaut.Call)

For a multi-output call, find variables on the tape referring to elements of the output.
Example:

    import Umlaut: Tape, Input, mkcall

    make_tuple(x) = (x, x + 1)

    tape = Tape()
    x = push!(tape, Input(1.0))
    out = push!(tape, mkcall(make_tuple, x))
    y1 = push!(tape, mkcall(getfield, out, 1))
    y2 = push!(tape, mkcall(getfield, out, 2))

    @assert unpacked_vars(tape, tape[out]) == [y1, y2]

"""
function unpacked_vars(op::Call)
    @assert op.val isa Tuple "Can't unpack non-tuple output"
    tape = op.tape
    # out = V(op)
    vars = Any[nothing for _=1:length(op.val)]
    found = 0
    for id in op.id + 1:length(tape)
        cur = tape[V(id)]
        if (cur isa Call && cur.fn in (getfield, getindex) &&   # is getfield call
            cur.args[1] isa V && cur.args[1].id == op.id)       # with V(op) as the first argument
            # found unpacked var
            idx = cur.args[2]
            idx = idx isa V ? tape[idx].val : idx
            vars[idx] = V(cur)
            # in most cases unpacked vars go right after the multi-ouput,
            # no need to search further
            found += 1
            found == length(vars) && break
        end
    end
    return vars
end
