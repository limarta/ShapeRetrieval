struct OTShell
end
Flux.@functor OTShell
Flux.trainable(m::OTShell) = (diffusion_time = m.diffusion_time,)

function (shell::OTShell)
end