"""Major F1 regulation eras and rule-reset metadata."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RegulationChange:
    """Structured summary of a major technical or sporting reset."""

    year: int
    type: str
    power_unit: str = ""
    aerodynamics: str = ""
    weight: str = ""
    tyres: str = ""
    key_changes: list[str] = field(default_factory=list)
    winners: list[str] = field(default_factory=list)
    losers: list[str] = field(default_factory=list)
    performance_impact: str = ""
    known_advantages: dict[str, str] = field(default_factory=dict)


REGULATION_HISTORY: dict[int, RegulationChange] = {
    1961: RegulationChange(
        year=1961,
        type="major",
        power_unit="Engine capacity cut from 2.5L to 1.5L naturally aspirated.",
        aerodynamics="Minimal aero era; packaging and lightweight balance dominated.",
        weight="Smaller, lighter cars became the norm.",
        key_changes=["1.5L naturally aspirated formula", "Shift from front-engined to rear-engined dominance"],
        winners=["ferrari", "lotus"],
        losers=[],
        performance_impact="Accelerated the rear-engine revolution and rewarded efficient packaging.",
    ),
    1966: RegulationChange(
        year=1966,
        type="major",
        power_unit="Three-litre naturally aspirated engines introduced.",
        aerodynamics="Mechanical grip still dominated, but power returned to the center of competitiveness.",
        weight="Cars grew to handle larger engines and fuel loads.",
        key_changes=["3.0L naturally aspirated / 1.5L supercharged formula", "Power became decisive again"],
        winners=["brabham"],
        losers=[],
        performance_impact="Reset the field around engine supply quality and reliability.",
    ),
    1983: RegulationChange(
        year=1983,
        type="major",
        aerodynamics="Ground-effect skirts banned, flat bottoms mandated.",
        key_changes=["Flat-bottom cars", "End of sliding skirts", "Turbo power remained"],
        winners=["mclaren", "williams"],
        losers=[],
        performance_impact="Reduced pure underbody aero and shifted performance back toward wings and engine power.",
    ),
    1989: RegulationChange(
        year=1989,
        type="major",
        power_unit="Turbo engines banned; naturally aspirated 3.5L engines mandated.",
        aerodynamics="High-downforce late-1980s wing development continued.",
        key_changes=["Turbo era ended", "3.5L naturally aspirated formula"],
        winners=["mclaren"],
        losers=["ferrari"],
        performance_impact="Reset power-unit hierarchies and emphasized drivability.",
    ),
    1994: RegulationChange(
        year=1994,
        type="major",
        aerodynamics="Electronic driver aids removed, plank and aero restrictions tightened.",
        key_changes=["Traction control and active suspension banned", "Wooden plank introduced", "Refueling returned"],
        winners=["benetton", "williams"],
        losers=[],
        performance_impact="Reintroduced more mechanical variance and sharply raised setup sensitivity.",
    ),
    1998: RegulationChange(
        year=1998,
        type="major",
        aerodynamics="Cars narrowed and grooved tyres introduced to cut cornering speeds.",
        tyres="Grooved tyres replaced slicks.",
        key_changes=["Narrow-track chassis", "Grooved tyres", "Reworked aero proportions"],
        winners=["mclaren", "ferrari"],
        losers=[],
        performance_impact="Changed the balance between low-speed mechanical grip and high-speed aero load.",
    ),
    2005: RegulationChange(
        year=2005,
        type="major",
        aerodynamics="Front-wing and diffuser rules reduced downforce.",
        tyres="Single-race tyre rule briefly enforced.",
        key_changes=["Reduced aero load", "Engines required to last two weekends", "Single-race tyre management emphasis"],
        winners=["renault", "mclaren"],
        losers=["ferrari"],
        performance_impact="Rewarded tyre management and operational precision over single-lap peak pace.",
    ),
    2009: RegulationChange(
        year=2009,
        type="major",
        aerodynamics="Wide front wings, narrow rear wings, clean-sheet diffuser rules.",
        key_changes=["KERS introduced", "Double-diffuser loophole", "Slick tyres returned"],
        winners=["brawn", "red_bull"],
        losers=["mclaren", "ferrari"],
        performance_impact="One of the most disruptive aero resets in modern F1, heavily reshuffling the order.",
    ),
    2014: RegulationChange(
        year=2014,
        type="major",
        power_unit="1.6L V6 turbo-hybrid power units replaced naturally aspirated V8s.",
        aerodynamics="Narrower noses and rebalanced aero surfaces.",
        weight="Cars grew heavier to accommodate hybrid systems.",
        key_changes=["Turbo-hybrid era began", "Fuel-flow limits", "ERS became strategically central"],
        winners=["mercedes"],
        losers=["renault"],
        performance_impact="Mercedes established the defining power-unit advantage of the era.",
    ),
    2017: RegulationChange(
        year=2017,
        type="major",
        aerodynamics="Wider cars, wider tyres and much higher downforce.",
        tyres="Substantially wider tyre footprint.",
        key_changes=["Wider front and rear wings", "Wider tyres", "Faster cornering speeds"],
        winners=["mercedes", "ferrari"],
        losers=[],
        performance_impact="Peak-aero machinery rewarded high-speed stability and clean-air dominance.",
    ),
    2022: RegulationChange(
        year=2022,
        type="major",
        aerodynamics="Ground effect return with simplified upper-body aero.",
        tyres="18-inch wheels introduced.",
        key_changes=["18-inch wheels", "Venturi tunnels", "Simplified front wings"],
        winners=["red_bull", "ferrari"],
        losers=["mercedes"],
        performance_impact="Red Bull and Ferrari adapted fastest; Mercedes lost early ground through porpoising.",
    ),
    2026: RegulationChange(
        year=2026,
        type="major",
        power_unit="New hybrid architecture with materially more ERS deployment and reduced ICE share.",
        aerodynamics="Active aerodynamics with moveable bodywork and reduced drag on straights.",
        weight="Cars target a lower minimum mass than late 2020s designs.",
        key_changes=[
            "Active aero on straights and in corners",
            "New 1.6L V6 hybrid with near 50/50 ICE-ERS balance",
            "Simplified front wing philosophy",
            "Ground effect concept retained",
        ],
        winners=["mercedes", "ferrari"],
        losers=[],
        performance_impact="Competitive order is expected to reshuffle meaningfully because both aero and power-unit architectures reset together.",
        known_advantages={
            "mercedes": "Historically strong hybrid integration and systems engineering.",
            "ferrari": "Longstanding power-unit and energy-recovery depth.",
        },
    ),
}
