import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
type Any = Record<string, any>;
type Workspace =
  "Race" | "Weekend" | "Championship" | "Live / replay" | "Evidence";
const pct = (v: number | undefined) =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;
const num = (v: number | undefined, d = 1) => (v == null ? "—" : v.toFixed(d));
const label = (s: string) => s.replaceAll("_", " ");
async function api(path: string, body?: unknown) {
  const response = await fetch(
    "/api/v1" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok)
    throw Error(
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail),
    );
  return result;
}
function Chart({
  data,
  layout = {},
  height = 330,
}: {
  data: any[];
  layout?: Any;
  height?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    let active = true;
    let plot: any;
    import("plotly.js-dist-min").then(({ default: P }) => {
      plot = P;
      if (active && ref.current)
        P.react(
          ref.current,
          data,
          {
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { family: "Inter, system-ui", color: "#93a2b5", size: 11 },
            margin: { t: 20, b: 40, l: 70, r: 20 },
            height,
            xaxis: { gridcolor: "#222b38", zeroline: false },
            yaxis: { gridcolor: "#222b38", zeroline: false },
            ...layout,
          },
          { responsive: true, displayModeBar: false },
        );
    });
    const el = ref.current;
    return () => {
      active = false;
      if (el && plot) plot.purge(el);
    };
  }, [data, layout, height]);
  return (
    <div
      ref={ref}
      className="plot"
      role="img"
      aria-label={layout.title || "Forecast chart"}
    />
  );
}
function Card({
  title,
  tag,
  children,
}: {
  title: string;
  tag?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card">
      <div className="card-title">
        <h2>{title}</h2>
        {tag && <span className="kind">{tag}</span>}
      </div>
      {children}
    </section>
  );
}
function Detail({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}
function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="empty">
      <span className="empty-mark">◌</span>
      {children}
    </div>
  );
}
function App() {
  const [tab, setTab] = useState<Workspace>("Race"),
    [health, setHealth] = useState<Any | null>(null),
    [events, setEvents] = useState<Any[]>([]),
    [event, setEvent] = useState(""),
    [stage, setStage] = useState("pre_weekend"),
    [cutoff, setCutoff] = useState(""),
    [forecast, setForecast] = useState<Any | null>(null),
    [season, setSeason] = useState<Any | null>(null),
    [frame, setFrame] = useState<Any | null>(null),
    [evidence, setEvidence] = useState<Any[]>([]),
    [ledger, setLedger] = useState<Any[]>([]),
    [jobs, setJobs] = useState<Any[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(""),
    [driver, setDriver] = useState(""),
    [opponent, setOpponent] = useState(""),
    [model, setModel] = useState("baseline:rating"),
    [syncYear, setSyncYear] = useState(2026),
    [messages, setMessages] = useState<Any[]>([]),
    [cursor, setCursor] = useState(20),
    [replayState, setReplayState] = useState<Any | null>(null),
    [scenario, setScenario] = useState(false),
    [strength, setStrength] = useState(0),
    [source, setSource] = useState<Any | null>(null),
    [prior, setPrior] = useState<Any | null>(null);
  const selected = events.find((e) => e.event_id === event);
  const isDemo = health?.illustrative;
  async function refresh() {
    const [h, ev, le, en, jo] = await Promise.all([
      api("/health"),
      api("/events"),
      api("/forecasts"),
      api("/evidence"),
      api("/jobs"),
    ]);
    setHealth(h);
    setEvents(ev);
    setLedger(le);
    setEvidence(en);
    setJobs(jo);
    return { h, ev };
  }
  async function act(name: string, fn: () => Promise<void>) {
    setBusy(name);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy("");
    }
  }
  useEffect(() => {
    act("Loading", async () => {
      const { h, ev } = await refresh();
      const next = h.illustrative
        ? ev.find((e: Any) => e.event_id === "2026-03")
        : ev.find((e: Any) => Date.parse(e.start_at) > Date.now()) || ev.at(-1);
      if (next) {
        setEvent(next.event_id);
        setCutoff(
          h.illustrative
            ? "2026-04-05T12:00"
            : new Date(
                Math.min(Date.now(), Date.parse(next.start_at) - 3600000),
              )
                .toISOString()
                .slice(0, 16),
        );
      }
    });
    const stream = new EventSource("/api/v1/jobs/stream");
    stream.onmessage = (e) => setJobs(JSON.parse(e.data));
    return () => stream.close();
  }, []);
  function request() {
    if (!cutoff) throw Error("Set an information cutoff");
    return {
      event_id: event,
      stage,
      as_of_utc: cutoff + ":00Z",
      model_version: model,
      simulations: 10000,
      seed: 42,
      ...(scenario
        ? { scenario: { strength_delta: driver ? { [driver]: strength } : {} } }
        : {}),
    };
  }
  async function generate() {
    await act("Simulating", async () => {
      const req = request();
      const [f, features] = await Promise.all([
        api("/forecasts", req),
        api("/features", req),
      ]);
      setPrior(forecast);
      setForecast(f);
      setFrame(features);
      setDriver(f.drivers[0].driver_id);
      setOpponent(f.drivers[1]?.driver_id || "");
      await refresh();
    });
  }
  async function enqueue(kind: string, parameters: Any) {
    await act("Starting job", async () => {
      await api("/jobs", { kind, parameters });
      await refresh();
    });
  }
  const drivers = forecast?.drivers || [];
  const focus = drivers.find((d: Any) => d.driver_id === driver) || drivers[0];
  const mismatch =
    !!forecast &&
    (forecast.request.event_id !== event ||
      forecast.request.stage !== stage ||
      forecast.request.as_of_utc.slice(0, 16) !== cutoff ||
      forecast.data_version !== health?.data_version);
  const evalEvidence = evidence.find(
    (e) => e.kind === "holdout_report" || e.kind === "development_experiment",
  );
  const modelReport =
    evalEvidence?.candidate || evalEvidence?.candidates?.[0]?.report;
  const frozen = evidence.find((e) => e.kind === "candidate_freeze");
  const last = ledger.find(
    (l) => l.kind === "race" && l.request.event_id === event,
  );
  function proof(f: Any) {
    return (
      <details>
        <summary>Forecast evidence & assumptions</summary>
        <div className="evidence-grid">
          <div>
            <b>Evidence class</b>
            <p>{label(f.evidence_kind)}</p>
            <b>Information cutoff</b>
            <p>{f.request.as_of_utc}</p>
            <b>Model</b>
            <p>{f.model_version || f.model_versions?.join(", ")}</p>
          </div>
          <div>
            <b>Versions & reproducibility</b>
            <p>Data: {f.data_version?.slice(0, 16)}</p>
            <p>Code: {f.code_version?.slice(0, 16)}</p>
            <p>Rules: {f.rules_version}</p>
            <p>
              Seed {f.seed} · {f.simulations?.toLocaleString()} samples{" "}
              <span className="kind">run configuration</span>
            </p>
          </div>
        </div>
        {f.missing_inputs?.map((m: string) => (
          <p className="notice" key={m}>
            {m}
          </p>
        ))}
        {f.assumptions?.map((a: string) => (
          <p key={a}>• {a}</p>
        ))}
        <p className="muted">
          Feature contributions explain model scores; they are not evidence of
          causation.
        </p>
        <div className="source-links">
          {f.source_ids?.map((s: string) => (
            <button
              key={s}
              onClick={() =>
                act("Reading source", async () =>
                  setSource(await api("/sources/" + s)),
                )
              }
            >
              Source {s.slice(0, 8)} ↗
            </button>
          ))}
        </div>
        <button
          onClick={() =>
            act("Reproducing", async () =>
              setSource(await api("/forecasts/" + f.id + "/reproduce", {})),
            )
          }
        >
          Verify reproduction
        </button>
        <details>
          <summary>Complete saved record</summary>
          <Detail value={f} />
        </details>
      </details>
    );
  }
  async function replay(predict = false) {
    await act("Replaying", async () => {
      const m = messages[cursor - 1];
      const result = await api("/replay", {
        event_id: m.event_id,
        as_of_utc: m.received_at,
        messages,
        predict,
      });
      setReplayState(result.state);
      if (result.forecast) {
        setPrior(forecast);
        setForecast(result.forecast);
        await refresh();
      }
    });
  }
  return (
    <div className="shell">
      <aside>
        <a className="brand" href="/" aria-label="F1 Engine home">
          <span className="brand-symbol">
            F<span>1</span>
          </span>
          <div>
            ENGINE<small>RACE INTELLIGENCE / V2</small>
          </div>
        </a>
        <div className="nav-caption">CONTROL ROOM</div>
        <nav>
          {(
            [
              "Race",
              "Weekend",
              "Championship",
              "Live / replay",
              "Evidence",
            ] as Workspace[]
          ).map((name, i) => (
            <button
              className={tab === name ? "active" : ""}
              key={name}
              onClick={() => setTab(name)}
            >
              <span className="nav-icon">{["◈", "▤", "◇", "◉", "▥"][i]}</span>
              {name}
              <span className="nav-num">0{i + 1}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div>
            <span className="dot" /> LOCAL ENGINE
          </div>
          <p>Evidence before confidence.</p>
          <small>
            {health?.ledger.valid
              ? "Ledger integrity checked"
              : "Checking integrity"}
          </small>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <div className="eyebrow">PERSONAL RACE CONTROL</div>
            <h1>
              {tab === "Race"
                ? "Every position. Every possibility."
                : tab === "Weekend"
                  ? "Read the weekend."
                  : tab === "Championship"
                    ? "The season, in perspective."
                    : tab === "Live / replay"
                      ? "Rewind. Observe. Reforecast."
                      : "Confidence needs evidence."}
            </h1>
          </div>
          <div className="connection">
            <span className="dot" />
            {isDemo ? "OFFLINE DEMO" : "LOCAL DATA"}
            <small>
              {isDemo ? "Fictional fixture dataset" : "Free public sources"}
            </small>
          </div>
        </header>
        {isDemo && (
          <div className="demo-banner">
            <b>ILLUSTRATIVE SCENARIO</b> Fictional drivers and results. These
            numbers demonstrate the application; they are not F1 forecasts or
            accuracy evidence.
          </div>
        )}
        <div className="controls">
          <label>
            Event <span className="kind">observation</span>
            <select
              aria-label="Event"
              value={event}
              onChange={(e) => {
                setEvent(e.target.value);
                const ev = events.find((v) => v.event_id === e.target.value);
                if (ev)
                  setCutoff(
                    new Date(
                      Math.min(Date.now(), Date.parse(ev.start_at) - 7200000),
                    )
                      .toISOString()
                      .slice(0, 16),
                  );
              }}
            >
              {events
                .slice()
                .reverse()
                .map((e) => (
                  <option key={e.event_id} value={e.event_id}>
                    {e.season} · {e.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            Stage
            <select
              aria-label="Stage"
              value={stage}
              onChange={(e) => setStage(e.target.value)}
            >
              {[
                "pre_weekend",
                "post_practice",
                "post_qualifying",
                "post_sprint",
              ].map((s) => (
                <option key={s} value={s}>
                  {label(s)}
                </option>
              ))}
            </select>
          </label>
          <label>
            Information cutoff · UTC
            <input
              aria-label="Information cutoff UTC"
              type="datetime-local"
              value={cutoff}
              onChange={(e) => setCutoff(e.target.value)}
            />
          </label>
          <button
            className="primary"
            disabled={!!busy || !event}
            onClick={generate}
          >
            {busy || "Generate forecast"} <span>↗</span>
          </button>
        </div>
        {error && (
          <div role="alert" className="error">
            <b>New result unavailable.</b> {error}{" "}
            {forecast && "The previous saved forecast remains below."}
          </div>
        )}
        {mismatch && (
          <div className="notice">
            Previous snapshot — the saved forecast does not match the current
            controls or dataset. Generate a new forecast to update it.
          </div>
        )}
        {!events.length && (
          <Card title="Start with evidence">
            <Empty>
              <h3>No dataset yet</h3>
              <p>
                Import one season at a time. Updates preserve all earlier
                seasons.
              </p>
              <p>
                For an offline preview, launch <code>f1engine app --demo</code>.
              </p>
              <label>
                Season
                <input
                  aria-label="Import season"
                  type="number"
                  min="2014"
                  max="2026"
                  value={syncYear}
                  onChange={(e) => setSyncYear(+e.target.value)}
                />
              </label>
              <button
                onClick={() => enqueue("sync_season", { season: syncYear })}
              >
                Import season
              </button>
            </Empty>
          </Card>
        )}
        {tab === "Race" && events.length > 0 && (
          <>
            <div className="section-heading">
              <h2>{selected?.name}</h2>
              <span className="muted">FULL-FIELD FORECAST</span>
            </div>
            <div className="model-controls">
              <label>
                Model
                <select
                  aria-label="Model"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="baseline:rating">
                    Driver / team rating baseline
                  </option>
                  <option value="baseline:grid">Grid baseline</option>
                  <option value="baseline:form">Recent-form baseline</option>
                  <option value="active">
                    Promoted model, or explicit baseline
                  </option>
                </select>
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={scenario}
                  onChange={(e) => setScenario(e.target.checked)}
                />
                Scenario experiment
              </label>
              {scenario && (
                <label>
                  Strength change for selected driver{" "}
                  <span className="kind">user assumption</span>
                  <input
                    type="number"
                    min="-3"
                    max="3"
                    step="0.1"
                    value={strength}
                    onChange={(e) => setStrength(+e.target.value)}
                  />
                </label>
              )}
            </div>
            {!forecast ? (
              <Card title="A forecast starts with a cutoff">
                <Empty>
                  <h3>Ready when you are</h3>
                  <p>
                    Generate the whole-field distribution from information
                    available at your selected time.
                  </p>
                  {last && (
                    <button
                      onClick={() => {
                        setForecast(last);
                        setDriver(last.drivers[0].driver_id);
                      }}
                    >
                      Open last saved forecast
                    </button>
                  )}
                </Empty>
              </Card>
            ) : (
              <>
                <div className="stats">
                  <div>
                    <span>HIGHEST WIN PROBABILITY</span>
                    <strong>
                      {
                        [...drivers].sort(
                          (a, b) => b.win_probability - a.win_probability,
                        )[0].name
                      }
                    </strong>
                    <small>
                      {pct(
                        Math.max(...drivers.map((d: Any) => d.win_probability)),
                      )}{" "}
                      · model estimate
                    </small>
                  </div>
                  <div>
                    <span>SIMULATED OUTCOMES</span>
                    <strong>{forecast.simulations.toLocaleString()}</strong>
                    <small>Run configuration · seed {forecast.seed}</small>
                  </div>
                  <div>
                    <span>FORECAST STATUS</span>
                    <strong>
                      {forecast.uncertainty.calibrated
                        ? "Calibrated candidate"
                        : "Baseline"}
                    </strong>
                    <small>{label(forecast.evidence_kind)}</small>
                  </div>
                </div>
                <div className="two-columns">
                  <Card title="Finishing distribution" tag="model estimate">
                    <Chart
                      data={[
                        {
                          type: "heatmap",
                          z: drivers.map((d: Any) => d.position_probabilities),
                          y: drivers.map((d: Any) => d.code),
                          x: drivers.map((_: Any, i: number) => i + 1),
                          colorscale: [
                            [0, "#121a25"],
                            [0.1, "#153e49"],
                            [0.3, "#287c86"],
                            [0.6, "#70cebd"],
                            [1, "#e6ffc5"],
                          ],
                          showscale: false,
                          hovertemplate:
                            "%{y} · P%{x}<br>%{z:.1%}<extra></extra>",
                        },
                      ]}
                      layout={{
                        title: "Probability by finishing position",
                        yaxis: { autorange: "reversed", type: "category" },
                        xaxis: {
                          title: { text: "Finishing position" },
                          dtick: 2,
                        },
                      }}
                      height={450}
                    />
                  </Card>
                  <Card title="Driver lens" tag="model estimate">
                    <select
                      aria-label="Driver"
                      value={driver}
                      onChange={(e) => setDriver(e.target.value)}
                    >
                      {drivers.map((d: Any) => (
                        <option value={d.driver_id} key={d.driver_id}>
                          {d.name}
                        </option>
                      ))}
                    </select>
                    {focus && (
                      <>
                        <div className="driver-head">
                          <span>{focus.team_name}</span>
                          <h3>{focus.name}</h3>
                          <strong>
                            {pct(focus.win_probability)}
                            <small> chance to win</small>
                          </strong>
                        </div>
                        <div className="mini-stats">
                          <p>
                            <b>{pct(focus.podium_probability)}</b>Podium
                          </p>
                          <p>
                            <b>{pct(focus.points_probability)}</b>Points
                          </p>
                          <p>
                            <b>{pct(focus.retirement_probability)}</b>Retirement
                          </p>
                        </div>
                        <p className="muted">
                          80% outcome interval: P{focus.position_p10}–P
                          {focus.position_p90}. Win sampling error: ±
                          {pct(focus.win_sampling_se)} (one standard error).
                        </p>
                        <div className="factors">
                          {focus.top_factors.map((f: Any) => (
                            <div key={f.feature}>
                              <span>{label(f.feature)}</span>
                              <b>{num(f.contribution, 3)}</b>
                            </div>
                          ))}
                        </div>
                        <small className="muted">
                          Contributions to model strength, not causes or
                          probability changes.
                        </small>
                        <label>
                          Head-to-head
                          <select
                            aria-label="Opponent"
                            value={opponent}
                            onChange={(e) => setOpponent(e.target.value)}
                          >
                            {drivers
                              .filter(
                                (d: Any) => d.driver_id !== focus.driver_id,
                              )
                              .map((d: Any) => (
                                <option key={d.driver_id} value={d.driver_id}>
                                  {d.name}
                                </option>
                              ))}
                          </select>
                        </label>
                        <p>
                          {pct(
                            forecast.head_to_head[focus.driver_id]?.[opponent],
                          )}{" "}
                          probability of finishing ahead.
                        </p>
                      </>
                    )}
                  </Card>
                </div>
                <Card title="Whole field" tag="model estimates">
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Driver</th>
                          <th>Expected finish</th>
                          <th>Win</th>
                          <th>Podium</th>
                          <th>Points</th>
                          <th>Retirement</th>
                          <th>80% interval</th>
                          {prior?.event.event_id ===
                            forecast.event.event_id && <th>Win change</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {drivers.map((d: Any) => (
                          <tr
                            key={d.driver_id}
                            onClick={() => setDriver(d.driver_id)}
                          >
                            <td>
                              <b>{d.name}</b>
                              <small>{d.team_name}</small>
                            </td>
                            <td>{num(d.expected_finish)}</td>
                            <td>{pct(d.win_probability)}</td>
                            <td>{pct(d.podium_probability)}</td>
                            <td>{pct(d.points_probability)}</td>
                            <td>{pct(d.retirement_probability)}</td>
                            <td>
                              P{d.position_p10}–P{d.position_p90}
                            </td>
                            {prior?.event.event_id ===
                              forecast.event.event_id && (
                              <td>
                                {num(
                                  100 *
                                    (d.win_probability -
                                      (prior?.drivers.find(
                                        (p: Any) => p.driver_id === d.driver_id,
                                      )?.win_probability || 0)),
                                  1,
                                )}{" "}
                                pp
                              </td>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {proof(forecast)}
                </Card>
              </>
            )}
          </>
        )}
        {tab === "Weekend" && (
          <>
            <div className="section-heading">
              <h2>Weekend evidence</h2>
              <button
                disabled={!event || !!busy}
                onClick={() =>
                  act("Loading weekend", async () =>
                    setFrame(await api("/features", request())),
                  )
                }
              >
                Inspect current cutoff
              </button>
            </div>
            {!frame ? (
              <Card title="Session coverage">
                <Empty>
                  Inspect the selected cutoff to see which inputs are available.
                </Empty>
              </Card>
            ) : (
              <>
                <div className="two-columns">
                  <Card title="Practice pace" tag="derived calculation">
                    <Chart
                      data={[
                        {
                          type: "bar",
                          orientation: "h",
                          x: frame.rows.map((r: Any) =>
                            r.practice_relative_pace == null
                              ? null
                              : 100 * r.practice_relative_pace,
                          ),
                          y: frame.rows.map((r: Any) => r.code),
                          marker: { color: "#78d7be" },
                        },
                      ]}
                      layout={{
                        title: "Relative pace within comparable lap groups (%)",
                        yaxis: { autorange: "reversed" },
                      }}
                    />
                    <p className="muted">
                      {frame.practice_filter.eligible_laps} eligible /{" "}
                      {frame.practice_filter.received_laps} received laps. Pit
                      and neutralized laps excluded. Fuel load unknown.
                    </p>
                  </Card>
                  <Card title="Input coverage" tag="derived calculation">
                    {Object.entries(frame.coverage).map(([k, v]) => (
                      <div className="coverage" key={k}>
                        <span>{label(k)}</span>
                        <progress max="1" value={v as number} />
                        <b>{pct(v as number)}</b>
                      </div>
                    ))}
                    {frame.missing.map((m: string) => (
                      <p className="notice" key={m}>
                        {m}
                      </p>
                    ))}
                  </Card>
                </div>
                <Card
                  title="Qualifying, sprint & weather"
                  tag={
                    isDemo
                      ? "illustrative scenario"
                      : "observations / derived calculations"
                  }
                >
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Driver</th>
                          <th>Qualifying rank</th>
                          <th>Qualifying gap (%)</th>
                          <th>Sprint points</th>
                          <th>Rain / climate prior</th>
                        </tr>
                      </thead>
                      <tbody>
                        {frame.rows.map((r: Any) => (
                          <tr key={r.driver_id}>
                            <td>{r.name}</td>
                            <td>{r.grid ?? "—"}</td>
                            <td>
                              {r.qualifying_gap == null
                                ? "—"
                                : num(r.qualifying_gap * 100, 3)}
                            </td>
                            <td>{r.sprint_points ?? "—"}</td>
                            <td>
                              {pct(r.rain_probability)}
                              <small>{label(r.weather_basis)}</small>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="muted">
                    Issued forecasts only. Historical observed weather is never
                    substituted.
                  </p>
                </Card>
              </>
            )}
          </>
        )}
        {tab === "Championship" && (
          <>
            <div className="section-heading">
              <h2>Season outlook</h2>
              <button
                className="primary"
                disabled={!event || !!busy}
                onClick={() =>
                  act("Simulating season", async () => {
                    setSeason(
                      await api("/championship", {
                        season: selected?.season,
                        as_of_utc: cutoff + ":00Z",
                        simulations: 10000,
                        seed: 42,
                      }),
                    );
                    await refresh();
                  })
                }
              >
                Simulate championship ↗
              </button>
            </div>
            {!season ? (
              <Card title="Every remaining round">
                <Empty>
                  Begin with official standings at the cutoff. Simulate
                  qualifying uncertainty, Grands Prix, sprints and championship
                  countback.
                </Empty>
              </Card>
            ) : (
              <>
                <div className="two-columns">
                  <Card title="Drivers’ championship" tag="model estimate">
                    <Chart
                      data={[
                        {
                          type: "bar",
                          orientation: "h",
                          x: season.drivers.map(
                            (d: Any) => d.title_probability,
                          ),
                          y: season.drivers.map((d: Any) => d.driver_id),
                          marker: { color: "#78d7be" },
                        },
                      ]}
                      layout={{
                        title: "Title probabilities",
                        yaxis: { autorange: "reversed" },
                        xaxis: { tickformat: ".0%" },
                      }}
                      height={430}
                    />
                  </Card>
                  <Card title="Constructors’ championship" tag="model estimate">
                    <Chart
                      data={[
                        {
                          type: "bar",
                          orientation: "h",
                          x: season.constructors.map(
                            (d: Any) => d.title_probability,
                          ),
                          y: season.constructors.map((d: Any) => d.team_id),
                          marker: { color: "#a6bafc" },
                        },
                      ]}
                      layout={{
                        title: "Title probabilities",
                        yaxis: { autorange: "reversed" },
                        xaxis: { tickformat: ".0%" },
                      }}
                      height={430}
                    />
                  </Card>
                </div>
                <Card title="Points outlook" tag="observation + model estimate">
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Driver</th>
                          <th>Current points</th>
                          <th>Expected points</th>
                          <th>80% points interval</th>
                          <th>Title</th>
                          <th>Elimination bound</th>
                        </tr>
                      </thead>
                      <tbody>
                        {season.drivers.map((d: Any) => (
                          <tr key={d.driver_id}>
                            <td>{d.driver_id}</td>
                            <td>{d.current_points}</td>
                            <td>{num(d.expected_points)}</td>
                            <td>
                              {num(d.points_p10, 0)}–{num(d.points_p90, 0)}
                            </td>
                            <td>{pct(d.title_probability)}</td>
                            <td>
                              {d.eliminated ? "Eliminated" : "Not ruled out"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {proof(season)}
                </Card>
              </>
            )}
          </>
        )}
        {tab === "Live / replay" && (
          <>
            <div className="section-heading">
              <h2>Deterministic replay</h2>
              <span className="badge">
                {replayState?.stale ? "STALE FEED" : "REPLAY MODE"}
              </span>
            </div>
            <Card title="Observation timeline" tag="arrival-time cutoff">
              <div className="inline">
                <label className="file">
                  Import normalized recording
                  <input
                    aria-label="Import replay recording"
                    type="file"
                    accept=".json"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f)
                        act("Reading recording", async () => {
                          const data = JSON.parse(await f.text());
                          if (!Array.isArray(data))
                            throw Error(
                              "Recording must be an array of replay envelopes",
                            );
                          setMessages(data);
                          setCursor(Math.min(20, data.length));
                        });
                    }}
                  />
                </label>
                {isDemo && (
                  <button
                    onClick={() =>
                      act("Loading fixture", async () => {
                        setMessages(await api("/replay/demo"));
                        setCursor(20);
                      })
                    }
                  >
                    Load illustrative replay
                  </button>
                )}
              </div>
              {messages.length > 0 && (
                <>
                  <input
                    aria-label="Replay position"
                    type="range"
                    min="1"
                    max={messages.length}
                    value={cursor}
                    onChange={(e) => setCursor(+e.target.value)}
                  />
                  <p>
                    {cursor} / {messages.length} received messages ·{" "}
                    {messages[cursor - 1]?.received_at}
                  </p>
                  <button onClick={() => replay()}>Inspect at cutoff</button>{" "}
                  <button onClick={() => replay(true)}>
                    Forecast remaining race
                  </button>
                </>
              )}
              <p className="muted">
                Future messages excluded. Revisions apply when received. OpenF1
                paid live access is not required.
              </p>
            </Card>
            {replayState && (
              <div className="two-columns">
                <Card
                  title="Timing tower"
                  tag={isDemo ? "illustrative scenario" : "observations"}
                >
                  <div className="table-scroll">
                    <table>
                      <thead>
                        <tr>
                          <th>Pos</th>
                          <th>Driver</th>
                          <th>Lap</th>
                          <th>Gap</th>
                          <th>Tyre</th>
                          <th>Age</th>
                        </tr>
                      </thead>
                      <tbody>
                        {replayState.drivers.map((r: Any) => (
                          <tr key={r.driver_id}>
                            <td>{r.position}</td>
                            <td>{r.driver_id}</td>
                            <td>{r.lap}</td>
                            <td>
                              {r.position === 1
                                ? "LEADER"
                                : r.gap_seconds == null
                                  ? "Lap difference"
                                  : num(r.gap_seconds) + "s"}
                            </td>
                            <td>{r.compound || "—"}</td>
                            <td>{r.tyre_age ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
                <Card title="Track & feed">
                  {replayState.drivers.some(
                    (r: Any) => r.x != null && r.y != null,
                  ) ? (
                    <Chart
                      data={[
                        {
                          type: "scatter",
                          mode: "markers+text",
                          x: replayState.drivers.map((r: Any) => r.x),
                          y: replayState.drivers.map((r: Any) => r.y),
                          text: replayState.drivers.map(
                            (r: Any) => r.driver_id,
                          ),
                          textposition: "top center",
                          marker: { color: "#b7f497", size: 9 },
                        },
                      ]}
                      layout={{
                        title: "Observed track positions",
                        yaxis: { scaleanchor: "x" },
                        showlegend: false,
                      }}
                    />
                  ) : (
                    <Empty>
                      <h3>Track position unavailable</h3>
                      <p>
                        This recording contains lap timing, not verified
                        position coordinates.
                      </p>
                    </Empty>
                  )}
                  <p>
                    Status: {replayState.status} · Age:{" "}
                    {replayState.age_seconds}s
                  </p>
                  <p className="muted">
                    State checksum: {replayState.visible_hash.slice(0, 20)}
                  </p>
                  <p className="muted">
                    Remaining-race forecasts appear in Race after generation.
                  </p>
                </Card>
              </div>
            )}
          </>
        )}
        {tab === "Evidence" && (
          <>
            <div className="section-heading">
              <h2>Evidence registry</h2>
              <button
                onClick={() =>
                  act("Refreshing evidence", async () => {
                    await refresh();
                  })
                }
              >
                Refresh evidence
              </button>
            </div>
            <div className="three-columns">
              {[
                "historical_reconstruction",
                "held_out_evaluation",
                "prospective_forecast",
              ].map((kind) => (
                <Card title={label(kind)} tag="separate evidence" key={kind}>
                  <strong className="big-number">
                    {ledger.filter((f) => f.evidence_kind === kind).length}
                  </strong>
                  <p>Saved forecasts in the latest {ledger.length} records</p>
                  <p className="muted">
                    {kind === "historical_reconstruction"
                      ? "Later reconstruction; publication-time limits apply."
                      : kind === "held_out_evaluation"
                        ? "Candidate and selection process must be frozen first."
                        : "Recorded before the event; scored once results arrive."}
                  </p>
                </Card>
              ))}
            </div>
            <Card title="Model comparisons & calibration">
              {modelReport && (
                <>
                  <Chart
                    data={[
                      {
                        type: "scatter",
                        mode: "lines",
                        x: [0, 1],
                        y: [0, 1],
                        line: { color: "#526578", dash: "dot" },
                        name: "Perfect calibration",
                      },
                      {
                        type: "scatter",
                        mode: "lines+markers",
                        x: modelReport.calibration.win.map(
                          (b: Any) => b.predicted,
                        ),
                        y: modelReport.calibration.win.map(
                          (b: Any) => b.observed,
                        ),
                        marker: { color: "#b7f497" },
                        name: "Winner reliability",
                      },
                    ]}
                    layout={{
                      title: "Calibration · empirical evaluation",
                      xaxis: {
                        title: { text: "Predicted frequency" },
                        tickformat: ".0%",
                      },
                      yaxis: {
                        title: { text: "Observed frequency" },
                        tickformat: ".0%",
                      },
                    }}
                  />
                  <p>
                    {modelReport.races} races · RPS{" "}
                    {num(modelReport.metrics.rps, 4)} · winner log loss{" "}
                    {num(modelReport.metrics.winner_log_loss, 4)}{" "}
                    <span className="kind">derived calculation</span>
                  </p>
                </>
              )}
              <p className="muted">
                Promotion requires a paired race-bootstrap improvement interval
                and no more than 2% deterioration in winner log loss.
              </p>
              {evidence.length ? (
                evidence.map((e) => (
                  <details key={e.id}>
                    <summary>
                      {label(e.kind || e.type || "evidence")} ·{" "}
                      {e.created_at?.slice(0, 10)}
                    </summary>
                    <Detail value={e} />
                  </details>
                ))
              ) : (
                <Empty>
                  No evaluation results yet. The baseline remains active.
                </Empty>
              )}
              <div className="inline">
                <button
                  disabled={!!busy}
                  onClick={() => enqueue("experiment", {})}
                >
                  Run bounded development experiment
                </button>
                {frozen && (
                  <button
                    disabled={!!busy}
                    onClick={() => enqueue("holdout", { freeze_id: frozen.id })}
                  >
                    Evaluate frozen candidate on 2025
                  </button>
                )}
                <button disabled={!!busy} onClick={() => enqueue("score", {})}>
                  Score saved forecasts
                </button>
                <button
                  disabled={!!busy}
                  onClick={() => enqueue("benchmark", {})}
                >
                  Measure cached performance
                </button>
              </div>
            </Card>
            <Card
              title="Append-only forecast ledger"
              tag={
                health?.ledger.valid
                  ? "integrity verified"
                  : "verification pending"
              }
            >
              {ledger.length ? (
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Saved at</th>
                        <th>Event</th>
                        <th>Evidence</th>
                        <th>Model</th>
                        <th>Record</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ledger.map((f) => (
                        <tr key={f.id}>
                          <td>{f.created_at.slice(0, 19)}</td>
                          <td>{f.event?.name || f.season}</td>
                          <td>{label(f.evidence_kind)}</td>
                          <td>{f.model_version || "season simulation"}</td>
                          <td>
                            <button onClick={() => setSource(f)}>
                              Inspect {f.id.slice(0, 8)}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <Empty>No saved forecasts.</Empty>
              )}
              <p className="muted">
                Local content hashes are not an external timestamp attestation.
              </p>
            </Card>
            <Card title="Data & system health">
              <div className="inline">
                <label>
                  Season <span className="kind">user input</span>
                  <input
                    aria-label="Season to import"
                    type="number"
                    min="2014"
                    max="2026"
                    value={syncYear}
                    onChange={(e) => setSyncYear(+e.target.value)}
                  />
                </label>
                <button
                  disabled={isDemo || !!busy}
                  onClick={() => enqueue("sync_season", { season: syncYear })}
                >
                  Import / refresh season
                </button>
                <button
                  disabled={isDemo || !event || !!busy}
                  onClick={() => enqueue("weather", { event_id: event })}
                >
                  Capture issued weather
                </button>
                <button
                  disabled={isDemo || !event || !!busy}
                  onClick={() => enqueue("climate", { event_id: event })}
                >
                  Capture climate prior
                </button>
              </div>
              <p className="muted">
                45-minute jobs · 4 training threads · at most 12 candidates.
                Detailed-session jobs are available through the API.
              </p>
              <details>
                <summary>Inspect system health</summary>
                <Detail value={health} />
              </details>
            </Card>
          </>
        )}
        {jobs.length > 0 && (
          <Card title="Background jobs" tag="observed status">
            <div className="job-list">
              {jobs.slice(0, 8).map((j) => (
                <div key={j.id} className="job">
                  <div>
                    <b>{label(j.kind)}</b>
                    <span>
                      {j.status} · {j.message}
                    </span>
                    {j.error && <small className="error-text">{j.error}</small>}
                  </div>
                  <progress max="1" value={j.progress} />
                  {["queued", "running"].includes(j.status) ? (
                    <button
                      onClick={() =>
                        act("Cancelling", async () => {
                          await api("/jobs/" + j.id + "/cancel", {});
                          await refresh();
                        })
                      }
                    >
                      Cancel
                    </button>
                  ) : ["failed", "paused", "interrupted", "cancelled"].includes(
                      j.status,
                    ) ? (
                    <button
                      onClick={() =>
                        act("Resuming", async () => {
                          await api("/jobs/" + j.id + "/resume", {});
                          await refresh();
                        })
                      }
                    >
                      Resume
                    </button>
                  ) : (
                    <button onClick={() => setSource(j.result)}>Result</button>
                  )}
                </div>
              ))}
            </div>
          </Card>
        )}
        <footer>
          <span>
            F1 ENGINE <b>V2</b>
          </span>
          <p>Observe carefully. Forecast honestly. Verify everything.</p>
          <small>
            Independent personal project. No affiliation with Formula 1.
          </small>
        </footer>
      </main>
      {source && (
        <div className="modal-backdrop" onClick={() => setSource(null)}>
          <div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-label="Evidence detail"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="card-title">
              <h2>Evidence detail</h2>
              <button autoFocus onClick={() => setSource(null)}>
                Close ×
              </button>
            </div>
            <Detail value={source} />
          </div>
        </div>
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
