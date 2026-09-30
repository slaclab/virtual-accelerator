"""Registry tests that need no engine dependencies or lattice checkouts."""

import pytest

from lume.actions import WritableActionMixin

from virtual_accelerator.registry import (
    _CHAIN_ALIASES,
    _normalize,
    _resolve_handoffs,
    _resolve_spec,
    _route_chain_kwargs,
    _route_single_kwargs,
    _strip_overlapping_variables,
    common_handoff_points,
    get_model,
    list_handoff_points,
    list_models,
)
from virtual_accelerator.registry.models import MODELS


class TestDiscovery:
    def test_all_entries_listed(self):
        assert set(list_models()) == set(MODELS)

    def test_repr_is_aligned_table(self):
        text = repr(list_models())
        assert "impact_cu_inj" in text
        # header row + separators + one row per model + staged-chain block
        assert "facility" in text and "simulator" in text
        assert "start" in text and "end" in text
        assert "IMPACT" in text and "Bmad" in text and "Facet2" in text
        # A staged-chain row is rendered when both stages pass the filter, with
        # the inferred handoff spelled out so users can see which plane
        # get_model will pick when handoff_loc is not passed.
        assert "impact_cu_inj -> bmad_cu_hxr (handoff YAG03)" in text
        # Endpoints appear as their own columns rather than in the description.
        assert "CATHODE" in text and "YAG03" in text

    def test_repr_omits_chain_block_when_no_pair_survives_filter(self):
        # Only Bmad models remain -- neither standard chain has both stages Bmad.
        text = repr(list_models(simulator="bmad"))
        assert "->" not in text

    def test_filter_by_engine_and_facility(self):
        assert list(list_models(simulator="bmad")) == ["bmad_cu_hxr", "bmad_f2_elec"]
        assert list(list_models(facility="facet2")) == [
            "impact_f2e_inj",
            "surrogate_f2e_inj",
            "bmad_f2_elec",
        ]
        assert set(list_models(facility="lcls")) | set(
            list_models(facility="facet2")
        ) == set(MODELS)

    def test_handoff_points_are_lattice_ordered(self):
        diags = list_handoff_points("bmad_cu_hxr")
        assert diags.index("YAG02") < diags.index("YAG03") < diags.index("OTR2")

    def test_impact_lists_only_its_standard_extent(self):
        # Standard injector extent is cathode -> YAG03. Screens further downstream
        # are excluded: YAG01/OTR3 are commented out in the deck, OTR4 is past
        # stop_1 at z=16.5, and OTR1/OTR2 are past the standard handoff.
        diags = list_handoff_points("impact_cu_inj")
        assert diags == ("YAG02", "YAG03")
        for absent in ("YAG01", "OTR1", "OTR2", "OTR3", "OTR4"):
            assert absent not in diags

    def test_cathode_is_only_listed_where_it_is_usable(self):
        # The bmad models accept a cathode start; the injectors have a fixed start,
        # so listing it there would advertise something that cannot be passed.
        # FACET's element is CATHODEF, not CATHODE.
        assert "CATHODE" in list_handoff_points("bmad_cu_hxr")
        assert "CATHODEF" in list_handoff_points("bmad_f2_elec")
        for fixed in ("impact_cu_inj", "surrogate_cu_inj"):
            assert not {"CATHODE", "CATHODEF"} & set(list_handoff_points(fixed))

    def test_facet_handoff_is_restricted_to_pr10241(self):
        for inj in ("impact_f2e_inj", "surrogate_f2e_inj"):
            assert list_handoff_points(inj) == ("PR10241",)
            assert common_handoff_points(inj, "bmad_f2_elec") == ("PR10241",)


class TestEntryIntegrity:
    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_builder_is_importable_path(self, name):
        module_path, sep, func = MODELS[name].builder.partition(":")
        assert sep and module_path.startswith("virtual_accelerator.") and func

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_extent_params_are_declared(self, name):
        entry = MODELS[name]
        for param in (entry.start_param, entry.end_param):
            if param is not None:
                assert param in entry.params

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_shared_params_are_declared(self, name):
        entry = MODELS[name]
        assert entry.shared_params <= set(entry.params)

    @pytest.mark.parametrize("name", sorted(MODELS))
    def test_defaults_are_consistent(self, name):
        entry = MODELS[name]
        if entry.default_start and entry.start_param:
            assert entry.params[entry.start_param] == entry.default_start
        if entry.default_end and entry.end_param:
            assert entry.params[entry.end_param] == entry.default_end


class TestValidation:
    def test_unknown_model(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("bmad_does_not_exist")

    def test_rejects_unavailable_screen(self):
        with pytest.raises(ValueError, match="not an available end screen"):
            get_model("impact_cu_inj", end_ele="OTR4")

    def test_rejects_cross_facility_staging(self):
        with pytest.raises(ValueError, match="different facilities"):
            get_model(["impact_cu_inj", "bmad_f2_elec"], handoff_loc="YAG03")

    def test_rejects_impact_as_downstream_stage(self):
        with pytest.raises(ValueError, match="only start at the cathode"):
            get_model(["bmad_cu_hxr", "impact_cu_inj"], handoff_loc="OTR2")

    def test_rejects_start_ele_on_fixed_extent_model(self):
        with pytest.raises(ValueError, match="fixed start"):
            get_model("surrogate_cu_inj", start_ele="OTR2")

    def test_rejects_single_model_list(self):
        with pytest.raises(ValueError, match="at least two models"):
            get_model(["bmad_cu_hxr"])

    def test_rejects_wrong_handoff_count(self):
        with pytest.raises(ValueError, match="handoff location"):
            get_model(["surrogate_cu_inj", "bmad_cu_hxr"], handoff_loc=["OTR2", "OTR3"])

    def test_rejects_cathode_as_handoff(self):
        with pytest.raises(ValueError, match="nothing is upstream"):
            get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="CATHODE")

    def test_rejects_facet_cathode_as_handoff(self):
        # CATHODEF is FACET's cathode spelling -- same "nothing upstream"
        # invariant, different name.
        with pytest.raises(ValueError, match="nothing is upstream"):
            get_model(["impact_f2e_inj", "bmad_f2_elec"], handoff_loc="CATHODEF")

    def test_rejects_handoff_not_shared_by_both_stages(self):
        # OTR4 is past impact_cu_inj's stop at z=16.5, so it cannot hand off there.
        with pytest.raises(ValueError, match="not a shared handoff point"):
            get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="OTR4")


class TestSingleModelKwargs:
    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="not a parameter of"):
            _route_single_kwargs(MODELS["bmad_cu_hxr"], {"n_particle": 5})

    def test_known_kwarg_passes_through(self):
        routed = _route_single_kwargs(MODELS["bmad_cu_hxr"], {"track_beam": True})
        assert routed == {"track_beam": True}

    def test_stage_kwargs_rejected_for_single_model(self):
        with pytest.raises(ValueError, match="only meaningful for a chain"):
            get_model("bmad_cu_hxr", stage_kwargs={"bmad_cu_hxr": {"track_beam": True}})


class TestChainKwargs:
    """New nested-dict kwarg contract.

    Top-level kwargs are for shared params only (they broadcast to every stage
    that declares them). Stage-specific params go through ``stage_kwargs``,
    keyed by stage name.
    """

    def test_shared_param_broadcasts_to_every_declaring_stage(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["cheetah_cu_hxr"]]
        routed = _route_chain_kwargs(entries, None, {"n_particles": 42})
        assert routed == [{"n_particles": 42}, {"n_particles": 42}]

    def test_shared_param_skips_non_declaring_stage(self):
        # bmad_cu_hxr does not declare n_particles, so it is not sent there.
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_chain_kwargs(entries, None, {"n_particles": 42})
        assert routed == [{"n_particles": 42}, {}]

    def test_stage_kwargs_targets_one_stage(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_chain_kwargs(entries, {"bmad_cu_hxr": {"track_beam": True}}, {})
        assert routed == [{}, {"track_beam": True}]

    def test_stage_kwargs_accepts_end_ele_alias(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_chain_kwargs(
            entries,
            {
                "impact_cu_inj": {"end_ele": "YAG02"},
                "bmad_cu_hxr": {"end_ele": "TD11"},
            },
            {},
        )
        assert routed == [{"end_element": "YAG02"}, {"end_element": "TD11"}]

    def test_stage_specific_at_top_level_is_rejected(self):
        # track_beam is bmad-only. Flat use in a chain does not name a stage.
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="stage-specific"):
            _route_chain_kwargs(entries, None, {"track_beam": True})

    def test_stage_specific_error_names_a_valid_stage(self):
        # The error should point the user at the stage_kwargs form.
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="bmad_cu_hxr"):
            _route_chain_kwargs(entries, None, {"custom_beam_path": "x.h5"})

    def test_unknown_kwarg_at_top_level_is_rejected(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="stage-specific"):
            _route_chain_kwargs(entries, None, {"n_particle": 5})

    def test_stage_kwargs_rejects_shared_param(self):
        # n_particles is shared: divergence between stages would break the
        # physical invariant that the beam flows through them.
        entries = [MODELS["surrogate_cu_inj"], MODELS["cheetah_cu_hxr"]]
        with pytest.raises(ValueError, match="shared param"):
            _route_chain_kwargs(entries, {"cheetah_cu_hxr": {"n_particles": 7}}, {})

    def test_stage_kwargs_rejects_unknown_stage(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="not a stage of this model"):
            _route_chain_kwargs(entries, {"nope": {"track_beam": True}}, {})

    def test_stage_kwargs_rejects_unknown_param(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="not a parameter of"):
            _route_chain_kwargs(entries, {"bmad_cu_hxr": {"bogus": 1}}, {})

    def test_stage_kwargs_rejects_non_dict_value(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(TypeError, match="must be a dict"):
            _route_chain_kwargs(entries, {"bmad_cu_hxr": "not a dict"}, {})

    def test_stage_kwargs_rejects_start_ele_on_fixed_extent(self):
        # surrogate_cu_inj has no configurable extent -- start_ele/end_ele are
        # meaningless there and should error rather than silently no-op.
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        with pytest.raises(ValueError, match="fixed"):
            _route_chain_kwargs(
                entries, {"surrogate_cu_inj": {"start_ele": "CATHODE"}}, {}
            )


class TestChainAliases:
    def test_alias_resolves_to_stage_pair(self):
        names, alias = _resolve_spec("high_fidelity_cu_hxr_s2e")
        assert names == ["impact_cu_inj", "bmad_cu_hxr"]
        assert alias == "high_fidelity_cu_hxr_s2e"

    def test_every_alias_points_at_registered_stages(self):
        for alias, (upstream, downstream) in _CHAIN_ALIASES.items():
            assert upstream in MODELS, f"{alias} upstream {upstream!r} unregistered"
            assert downstream in MODELS, (
                f"{alias} downstream {downstream!r} unregistered"
            )

    def test_plain_name_returns_singleton_and_no_alias(self):
        names, alias = _resolve_spec("bmad_cu_hxr")
        assert names == ["bmad_cu_hxr"]
        assert alias is None

    def test_list_spec_is_returned_verbatim(self):
        names, alias = _resolve_spec(["impact_cu_inj", "bmad_cu_hxr"])
        assert names == ["impact_cu_inj", "bmad_cu_hxr"]
        assert alias is None

    def test_duplicate_stage_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate stage"):
            _resolve_spec(["bmad_cu_hxr", "bmad_cu_hxr"])

    def test_list_of_one_still_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            _resolve_spec(["bmad_cu_hxr"])


class TestHandoffResolution:
    def test_inferred_from_upstream_fixed_end(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        assert _resolve_handoffs(entries, None) == ["OTR2"]

    def test_explicit_handoff_is_used_verbatim(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        assert _resolve_handoffs(entries, "YAG03") == ["YAG03"]


class TestCommonHandoffPoints:
    def test_intersection_of_the_two_standard_chains(self):
        assert common_handoff_points("impact_cu_inj", "bmad_cu_hxr") == (
            "YAG02",
            "YAG03",
        )
        assert common_handoff_points("surrogate_cu_inj", "bmad_cu_hxr") == ("OTR2",)

    def test_cathode_is_always_excluded(self):
        # Both cathode spellings must be dropped, not just the LCLS one. FACET's
        # cathode carries an "F" suffix (CATHODEF); missing it would let a chain
        # hand off at the front of the machine, which has nothing upstream.
        assert "CATHODE" in MODELS["bmad_cu_hxr"].handoff_points
        assert "CATHODE" not in common_handoff_points("bmad_cu_hxr", "bmad_cu_hxr")
        assert "CATHODEF" in MODELS["bmad_f2_elec"].handoff_points
        assert "CATHODEF" not in common_handoff_points("bmad_f2_elec", "bmad_f2_elec")

    def test_is_intersection_not_union(self):
        # OTR4 is only reachable by bmad_cu_hxr; a union would wrongly include it.
        shared = common_handoff_points("impact_cu_inj", "bmad_cu_hxr")
        assert "OTR4" in MODELS["bmad_cu_hxr"].handoff_points
        assert "OTR4" not in shared

    def test_ordered_by_lattice_position(self):
        shared = common_handoff_points("impact_cu_inj", "bmad_cu_hxr")
        assert list(shared) == sorted(
            shared, key=MODELS["impact_cu_inj"].handoff_points.index
        )

    def test_no_shared_points_gives_empty_tuple(self):
        assert common_handoff_points("impact_cu_inj", "surrogate_cu_inj") == ()

    def test_cross_facility_pairs_share_nothing(self):
        assert common_handoff_points("impact_cu_inj", "bmad_f2_elec") == ()

    def test_requires_at_least_two_models(self):
        with pytest.raises(ValueError, match="at least two models"):
            common_handoff_points("bmad_cu_hxr")

    def test_unknown_model_name(self):
        with pytest.raises(KeyError, match="Unknown model"):
            common_handoff_points("impact_cu_inj", "nope")


class _FakeStage:
    """Minimal stand-in for a LUMEModel with registerable action variables."""

    def __init__(self, variables):
        self._vars = dict(variables)

    @property
    def supported_variables(self):
        return dict(self._vars)

    def unregister_action_variable(self, name):
        return self._vars.pop(name)


class _ReadOnlyVar:
    pass


class _WritableVar(WritableActionMixin):
    def _get(self, simulator):  # pragma: no cover - never invoked
        raise NotImplementedError

    def _set(self, simulator, value):  # pragma: no cover - never invoked
        raise NotImplementedError


class TestOverlapRemoval:
    """The handoff element belongs to both stages, so both publish its PVs.

    The upstream stage owns them since it tracks the beam to that plane, so they
    are unregistered downstream rather than moving the downstream start element.
    """

    def test_shared_read_only_variables_are_removed_downstream(self):
        up = _FakeStage({"SCREEN:IMAGE": _ReadOnlyVar(), "UP:ONLY": _ReadOnlyVar()})
        down = _FakeStage({"SCREEN:IMAGE": _ReadOnlyVar(), "DOWN:ONLY": _ReadOnlyVar()})
        removed = _strip_overlapping_variables(up, down, "up", "down")
        assert removed == ["SCREEN:IMAGE"]
        assert set(down.supported_variables) == {"DOWN:ONLY"}
        # the upstream stage keeps its copy
        assert "SCREEN:IMAGE" in up.supported_variables

    def test_no_overlap_is_a_no_op(self):
        up = _FakeStage({"UP:ONLY": _ReadOnlyVar()})
        down = _FakeStage({"DOWN:ONLY": _ReadOnlyVar()})
        assert _strip_overlapping_variables(up, down, "up", "down") == []
        assert set(down.supported_variables) == {"DOWN:ONLY"}

    def test_writable_overlap_raises_instead_of_silently_dropping(self):
        # Both stages driving the same magnet means the extents overlap rather
        # than meeting at a plane; dropping it downstream would leave that stage
        # tracking with a stale value.
        up = _FakeStage({"QUAD:BCTRL": _WritableVar()})
        down = _FakeStage({"QUAD:BCTRL": _WritableVar()})
        with pytest.raises(ValueError, match="writable variable"):
            _strip_overlapping_variables(up, down, "up", "down")
        assert "QUAD:BCTRL" in down.supported_variables

    def test_stage_without_unregister_support_raises(self):
        class Fixed:
            supported_variables = {"SHARED": _ReadOnlyVar()}

        up = _FakeStage({"SHARED": _ReadOnlyVar()})
        with pytest.raises(TypeError, match="unregister_action_variable"):
            _strip_overlapping_variables(up, Fixed(), "up", "down")


class TestStagedOverlap:
    """End-to-end coverage of ``get_model``'s staging block.

    Verifies the loop after ``_route_chain_kwargs`` that assembles stages, forces
    beam tracking on, applies the exclusive-end handoff, then hands consecutive
    stage pairs to ``_strip_overlapping_variables`` before wrapping in
    ``StagedModel``.
    """

    @staticmethod
    def _install(monkeypatch, stages_by_name):
        """Patch registry internals so ``get_model`` builds without a simulator.

        ``_build`` records its arguments in ``build_calls`` and returns the
        prepared ``_FakeStage`` for the requested entry; ``StagedModel`` is
        stubbed to a container that skips its own duplicate-variable
        validation, so the assertions can focus on the registry's own overlap
        handling and handoff wiring rather than reproducing StagedModel's checks.

        Returns
        -------
        list[dict]
            One entry per ``_build`` call in call order, with keys ``entry``,
            ``call_kwargs``, ``start_ele``, ``end_ele`` -- the exact arguments
            ``get_model`` passed for that stage.
        """
        import virtual_accelerator.registry as reg
        import lume.staged_model as staged_module

        build_calls: list[dict] = []

        def fake_build(entry, call_kwargs, start_ele, end_ele):
            build_calls.append(
                {
                    "entry": entry,
                    "call_kwargs": dict(call_kwargs),
                    "start_ele": start_ele,
                    "end_ele": end_ele,
                }
            )
            return stages_by_name[entry.name]

        class FakeStagedModel:
            def __init__(self, instances):
                self.lume_model_instances = list(instances)

            @property
            def supported_variables(self):
                return {
                    name: var
                    for stage in self.lume_model_instances
                    for name, var in stage.supported_variables.items()
                }

        monkeypatch.setattr(reg, "_build", fake_build)
        monkeypatch.setattr(staged_module, "StagedModel", FakeStagedModel)
        return build_calls

    def test_read_only_overlap_is_removed_downstream(self, monkeypatch):
        upstream = _FakeStage(
            {"YAG03:IMAGE": _ReadOnlyVar(), "IMPACT:ONLY": _ReadOnlyVar()}
        )
        downstream = _FakeStage(
            {"YAG03:IMAGE": _ReadOnlyVar(), "BMAD:ONLY": _ReadOnlyVar()}
        )
        self._install(
            monkeypatch,
            {"impact_cu_inj": upstream, "bmad_cu_hxr": downstream},
        )

        model = get_model(
            ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
        )

        assert set(model.supported_variables) == {
            "YAG03:IMAGE",
            "IMPACT:ONLY",
            "BMAD:ONLY",
        }
        # Duplicate lives only on the upstream stage after the strip.
        assert "YAG03:IMAGE" in upstream.supported_variables
        assert "YAG03:IMAGE" not in downstream.supported_variables

    def test_writable_overlap_raises_before_staged_model_is_built(self, monkeypatch):
        upstream = _FakeStage({"QUAD:BCTRL": _WritableVar()})
        downstream = _FakeStage({"QUAD:BCTRL": _WritableVar()})
        self._install(
            monkeypatch,
            {"impact_cu_inj": upstream, "bmad_cu_hxr": downstream},
        )

        with pytest.raises(ValueError, match="writable variable"):
            get_model(
                ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
            )
        # Nothing was mutated -- the downstream stage keeps its copy for retry.
        assert "QUAD:BCTRL" in downstream.supported_variables

    def test_no_overlap_leaves_both_stages_intact(self, monkeypatch):
        upstream = _FakeStage({"IMPACT:ONLY": _ReadOnlyVar()})
        downstream = _FakeStage({"BMAD:ONLY": _ReadOnlyVar()})
        self._install(
            monkeypatch,
            {"impact_cu_inj": upstream, "bmad_cu_hxr": downstream},
        )

        model = get_model(
            ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
        )

        assert set(upstream.supported_variables) == {"IMPACT:ONLY"}
        assert set(downstream.supported_variables) == {"BMAD:ONLY"}
        assert set(model.supported_variables) == {"IMPACT:ONLY", "BMAD:ONLY"}

    def test_stages_are_ordered_upstream_first(self, monkeypatch):
        upstream = _FakeStage({"IMPACT:ONLY": _ReadOnlyVar()})
        downstream = _FakeStage({"BMAD:ONLY": _ReadOnlyVar()})
        self._install(
            monkeypatch,
            {"impact_cu_inj": upstream, "bmad_cu_hxr": downstream},
        )

        model = get_model(
            ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
        )

        assert model.lume_model_instances == [upstream, downstream]


class TestStagedHandoffWiring:
    """The upstream stage must not include the handoff element.

    The registry engineers exclusivity per engine: Bmad gets the Tao
    ``"<handoff>-1"`` offset, IMPACT gets ``include_end_element=False``. Both
    paths need coverage -- without it, a regression that reverted either would
    show up only as a duplicate-variable failure inside ``StagedModel`` after a
    full simulator run.
    """

    def _stages(self):
        return {
            "impact_cu_inj": _FakeStage({"IMPACT:ONLY": _ReadOnlyVar()}),
            "surrogate_cu_inj": _FakeStage({"SURROGATE:ONLY": _ReadOnlyVar()}),
            "bmad_cu_hxr": _FakeStage({"BMAD:ONLY": _ReadOnlyVar()}),
        }

    def test_impact_upstream_gets_include_end_element_false(self, monkeypatch):
        # IMPACT's set_stop_location prunes to s <= stop, so without this flag
        # the boundary element (and its PVs) would stay on the upstream stage
        # and collide with the downstream stage that owns the plane.
        stages = self._stages()
        build_calls = TestStagedOverlap._install(monkeypatch, stages)

        get_model(
            ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
        )

        upstream_call = next(
            c for c in build_calls if c["entry"].name == "impact_cu_inj"
        )
        assert upstream_call["call_kwargs"].get("include_end_element") is False

    def test_impact_upstream_end_ele_is_bare_handoff(self, monkeypatch):
        # The exclusion happens via include_end_element=False, not via a "-1"
        # offset in the element name -- IMPACT has no such syntax.
        stages = self._stages()
        build_calls = TestStagedOverlap._install(monkeypatch, stages)

        get_model(
            ["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="YAG03", n_particles=100
        )

        upstream_call = next(
            c for c in build_calls if c["entry"].name == "impact_cu_inj"
        )
        assert upstream_call["end_ele"] == "YAG03"

    def test_bmad_upstream_end_ele_gets_minus_one_offset(self, monkeypatch):
        # Bmad's -slice_lattice accepts "<name>-1" to mean the element before
        # <name>, so the upstream slice ends immediately before the handoff.
        # This branch of _exclusive_end is only reachable when the upstream
        # engine is bmad -- covered here with a bmad -> bmad chain-like
        # arrangement using the surrogate as a placeholder is not possible, so
        # we exercise it directly on _exclusive_end.
        from virtual_accelerator.registry import _exclusive_end
        from virtual_accelerator.registry.models import MODELS

        assert _exclusive_end(MODELS["bmad_cu_hxr"], "OTR2") == "OTR2-1"

    def test_non_impact_upstream_omits_include_end_element(self, monkeypatch):
        # surrogate_cu_inj does not declare include_end_element, so the
        # registry must not set it -- the surrogate has a fixed extent that
        # already ends at OTR2, and injecting the flag would raise a
        # TypeError from the builder.
        stages = self._stages()
        build_calls = TestStagedOverlap._install(monkeypatch, stages)

        get_model(
            ["surrogate_cu_inj", "bmad_cu_hxr"], handoff_loc="OTR2", n_particles=100
        )

        upstream_call = next(
            c for c in build_calls if c["entry"].name == "surrogate_cu_inj"
        )
        assert "include_end_element" not in upstream_call["call_kwargs"]

    def test_downstream_stage_keeps_default_inclusive_end(self, monkeypatch):
        # The exclusion applies only to the *upstream* stage. The downstream
        # stage still owns the handoff plane, so nothing about it changes at
        # the boundary -- and its end_ele is the user-facing overall end.
        stages = self._stages()
        build_calls = TestStagedOverlap._install(monkeypatch, stages)

        get_model(
            ["impact_cu_inj", "bmad_cu_hxr"],
            handoff_loc="YAG03",
            end_ele="OTR4",
            n_particles=100,
        )

        downstream_call = next(
            c for c in build_calls if c["entry"].name == "bmad_cu_hxr"
        )
        assert "include_end_element" not in downstream_call["call_kwargs"]
        assert downstream_call["start_ele"] == "YAG03"
        assert downstream_call["end_ele"] == "OTR4"

    def test_single_model_impact_call_keeps_include_end_element_true(self, monkeypatch):
        # Single-model use is unchanged: get_model("impact_cu_inj",
        # end_ele="YAG03") should NOT set include_end_element=False, because
        # there is no downstream stage to own the plane. The user asked to
        # stop at YAG03 and expects YAG03's PVs.
        stages = self._stages()
        build_calls = TestStagedOverlap._install(monkeypatch, stages)

        get_model("impact_cu_inj", end_ele="YAG03", n_particles=100)

        (call,) = build_calls
        assert call["call_kwargs"].get("include_end_element", True) is True


class TestElementNameCase:
    """Element names are normalised at the API boundary.

    Tao is case-insensitive so lower case would appear to work, but IMPACT's
    impact.ele[...] is a dict lookup and the registry's own handoff_points and
    handoff_points lookups would silently miss.
    """

    @pytest.mark.parametrize("given", ["OTR4", "otr4", "Otr4", "oTr4"])
    def test_normalize_is_idempotent_upper(self, given):
        assert _normalize(given) == "OTR4"

    def test_normalize_passes_none_through(self):
        assert _normalize(None) is None

    def test_lowercase_bad_screen_is_still_rejected(self):
        # Before normalisation this slipped past validation and failed later
        # inside Tao with a far worse message.
        with pytest.raises(ValueError, match="not an available end screen"):
            get_model("impact_cu_inj", end_ele="otr99")

    def test_lowercase_valid_screen_is_accepted(self):
        # Validation must not reject a lowercase-but-valid screen. The builder may
        # succeed or fail depending on environment (extras, lattice env vars); the
        # only thing this test guards against is the "not an available" ValueError.
        try:
            get_model("impact_cu_inj", end_ele="yag03")
        except Exception as exc:
            assert "not an available" not in str(exc)

    def test_lowercase_handoff_normalises_before_resolution(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        handoffs = [_normalize(h) for h in _resolve_handoffs(entries, "yag03")]
        assert handoffs == ["YAG03"]
