from virtual_accelerator.surrogates.injector_surrogate import InjectorSurrogate

def test_injector_surrogate():
    # test to make sure that the surrogate can be 
    # initialized and returns an output beam distribution
    surrogate = InjectorSurrogate(n_particles=1000)
    output = surrogate.get(["output_beam"])
    assert "output_beam" in output
    beam = output["output_beam"]
    assert beam.x.shape[0] == 1000

    # check to make sure that changing a control variable changes 
    # the output beam distribution
    initial_beam = surrogate.get(["output_beam"])["output_beam"]
    surrogate.set({"QUAD:IN20:525:BCTRL": -5.0})
    updated_beam = surrogate.get(["output_beam"])["output_beam"]
    assert not (initial_beam.x == updated_beam.x).all()
    assert surrogate.get(["QUAD:IN20:525:BCTRL"])["QUAD:IN20:525:BCTRL"] == -5.0

