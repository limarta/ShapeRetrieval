struct Foo
    x::Matrix{Float64}
    y::Int
    function Foo(x,y)
        new()
        new.x = 1
        new.y = 1
    end
 end