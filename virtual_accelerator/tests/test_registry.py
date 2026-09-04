"""Registry tests that need no engine dependencies or lattice checkouts."""

import pytest

from lume.actions import WritableActionMixin

from virtual_accelerator.registry import (
    _normalize,
    _resolve_handoffs,
    _route_kwargs,
    _strip_overlapping_variables,
    common_handoff_points,
    get_model,
    list_handoff_points,
    list_models,
    models_available,
)
from virtual_accelerator.registry.models import MODELS


class TestDiscovery:
    def test_all_entries_listed(self):
        assert set(models_available) == set(MODELS)

    def test_repr_is_aligned_table(self):
        text = repr(models_available)
        assert "impact_cu_inj" in text
        assert len(text.splitlines()) == len(MODELS)

    def test_filter_by_engine_and_facility(self):
        assert list_models(engine="bmad") == ["bmad_cu_hxr", "bmad_f2_elec"]
        assert list_models(facility="facet2") == [
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
        for fixed in ("impact_cu_inj", "surrogate_cu_inj", "cheetah_cu_hxr"):
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

    def test_rejects_handoff_not_shared_by_both_stages(self):
        # OTR4 is past impact_cu_inj's stop at z=16.5, so it cannot hand off there.
        with pytest.raises(ValueError, match="not a shared handoff point"):
            get_model(["impact_cu_inj", "bmad_cu_hxr"], handoff_loc="OTR4")


class TestKwargRouting:
    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="not a parameter of any stage"):
            _route_kwargs([MODELS["bmad_cu_hxr"]], {"n_particle": 5})

    def test_shared_param_reaches_every_declaring_stage(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["cheetah_cu_hxr"]]
        routed = _route_kwargs(entries, {"n_particles": 42})
        assert routed == [{"n_particles": 42}, {"n_particles": 42}]

    def test_routes_to_single_declaring_stage(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_kwargs(entries, {"track_beam": True})
        assert routed == [{}, {"track_beam": True}]

    def test_dotted_form_targets_one_stage(self):
        entries = [MODELS["surrogate_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_kwargs(entries, {"bmad_cu_hxr.track_beam": True})
        assert routed == [{}, {"track_beam": True}]

    def test_dotted_form_rejects_unknown_stage(self):
        with pytest.raises(ValueError, match="not in this model"):
            _route_kwargs([MODELS["bmad_cu_hxr"]], {"nope.track_beam": True})

    def test_shared_param_cannot_be_set_per_stage(self):
        # n_particles must match across stages -- the beam flows through them.
        with pytest.raises(ValueError, match="same in every stage"):
            _route_kwargs(
                [MODELS["surrogate_cu_inj"], MODELS["cheetah_cu_hxr"]],
                {"cheetah_cu_hxr.n_particles": 7},
            )

    @pytest.mark.parametrize("key", ["end_element", "start_element"])
    def test_flat_builder_spelling_is_rejected(self, key):
        with pytest.raises(ValueError, match="does not say which stage"):
            _route_kwargs(
                [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]], {key: "TD11"}
            )

    def test_dotted_form_accepts_end_ele_per_stage(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        routed = _route_kwargs(
            entries, {"impact_cu_inj.end_ele": "YAG02", "bmad_cu_hxr.end_ele": "TD11"}
        )
        assert routed == [{"end_element": "YAG02"}, {"end_element": "TD11"}]

    def test_dotted_form_rejects_unknown_param(self):
        with pytest.raises(ValueError, match="not a parameter of"):
            _route_kwargs([MODELS["bmad_cu_hxr"]], {"bmad_cu_hxr.bogus": 1})


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
        # bmad lists CATHODE, so the intersection must drop it explicitly.
        assert "CATHODE" in MODELS["bmad_cu_hxr"].handoff_points
        assert "CATHODE" not in common_handoff_points("bmad_cu_hxr", "bmad_cu_hxr")

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
        assert common_handoff_points("cheetah_cu_hxr", "bmad_cu_hxr") == ()

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
        # Reaches the builder (and fails only because the extra is absent here),
        # proving validation no longer rejects it.
        with pytest.raises((ImportError, ValueError)) as excinfo:
            get_model("impact_cu_inj", end_ele="yag03")
        assert "not an available" not in str(excinfo.value)

    def test_lowercase_handoff_normalises_before_resolution(self):
        entries = [MODELS["impact_cu_inj"], MODELS["bmad_cu_hxr"]]
        handoffs = [_normalize(h) for h in _resolve_handoffs(entries, "yag03")]
        assert handoffs == ["YAG03"]
