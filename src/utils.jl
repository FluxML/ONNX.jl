"""
    rename_keys(d::Dict, subs::Dict)

Create a copy of `d`, replacing its keys according to mapping `subs`
"""
function rename_keys(dct::Dict, subs::Dict)
    new = empty(dct)
    for (key, val) in dct
        new_key = get(subs, key, key)
        new[new_key] = val
    end
    return new
end