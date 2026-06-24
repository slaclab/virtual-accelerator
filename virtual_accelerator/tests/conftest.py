def pytest_configure(config):
    """Register custom markers used for optional-backend test gating."""
    config.addinivalue_line(
        "markers",
        "requires_bmad: test requires bmad optional dependencies",
    )
    config.addinivalue_line(
        "markers",
        "requires_cheetah: test requires cheetah optional dependencies",
    )
    config.addinivalue_line(
        "markers",
        "requires_surrogate: test requires surrogate optional dependencies",
    )
    config.addinivalue_line(
        "markers",
        "requires_staged_model: test requires staged-model optional dependencies",
    )
    config.addinivalue_line(
        "markers",
        "requires_lcls_lattice: test requires LCLS_LATTICE env var",
    )
    config.addinivalue_line(
        "markers",
        "requires_facet2_lattice: test requires FACET2_LATTICE env var",
    )
