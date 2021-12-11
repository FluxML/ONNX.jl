"""
    rename_keys(d::Dict, subs::Dict)

Create a copy of `d`, replacing its keys according to mapping `subs`
"""
rename_keys(dct::Dict, subs::Dict) = Dict(get(subs, k, k) => v for (k, v) in pairs(dct))