import numpy as np
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sympy as sp
from sympy import symbols
import dash_split_pane
import json

# basic constants and parameters
c = 1  # c is dimensionless, normalized to 1
t_max = 100
default_resolution = 100
base_play_interval_ms = 50
play_interval_ms = base_play_interval_ms
time_steps = np.linspace(0, t_max, default_resolution)

# symbolic constant for c
c_sym = sp.Symbol('c', real=True)

# initial inertial frame at x(t)=0
frames = {
    "Static Frame": {"frame_type": "function", "variable": "x", "expr_str": "0", "display": True, "precomputed": None},
}

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])

def evaluate_motion_function(function_str, t_vals):
    # evaluate x(t) using given t array
    t = sp.symbols("t", real=True)
    expr = sp.sympify(function_str, locals={'c': c_sym, 't': t})
    expr = expr.subs(c_sym, c)
    return np.array([float(expr.subs(t, val)) for val in t_vals])

def parse_input_function(func_str):
    # parse input to classify as event or function frame
    # input must have both t=... and x=... for an event
    # no single-variable definitions allowed
    t = sp.Symbol('t', real=True)
    func_str = func_str.strip()

    # split by commas to detect t=..., x=...
    parts = [p.strip() for p in func_str.split(',')]
    assignments = {}
    for p in parts:
        if '=' in p:
            left, right = p.split('=', 1)
            var = left.strip()
            expr = right.strip()
            # reject any variable other than t or x
            if var not in ['x', 't']:
                return {"frame_type": "invalid", "reason": "only 'x' can be used to represent position; invalid variable encountered"}
            assignments[var] = expr

    has_t = 't' in assignments
    has_x = 'x' in assignments

    if has_t and has_x:
        # both t and x given, define an event
        # event must have no time-dependence
        try:
            t_val = float(sp.sympify(assignments['t'], locals={'c': c_sym, 't': t}))
            x_val = float(sp.sympify(assignments['x'], locals={'c': c_sym, 't': t}))
        except Exception as e:
            return {"frame_type": "invalid", "reason": f"invalid event: {str(e)}"}
        return {
            "frame_type": "event",
            "t_value": t_val,
            "x_value": x_val
        }

    if '=' in func_str and not (has_t and has_x):
        # x=... can define a function frame if time-dependent
        if has_t and not has_x:
            # single t=... is invalid
            return {"frame_type": "invalid", "reason": "`t=?` alone is not allowed. must provide both t and x for an event"}
        if has_x and not has_t:
            try:
                sp_expr = sp.sympify(assignments['x'], locals={'c': c_sym, 't': t})
                sp_expr.subs(t,0)
                v_expr = sp.diff(sp_expr, t)
                # frames with time-dependent velocity are non-inertial, reject them
                if t in v_expr.free_symbols:
                    return {"frame_type": "invalid", "reason": "non-inertial frame: velocity depends on t"}
                # ensure speed < c
                v = float(v_expr.subs(t, 0))
                if abs(v) >= 1:
                    return {"frame_type": "invalid", "reason": "speed >= c not allowed"}
                return {"frame_type": "function", "variable": "x", "expr_str": assignments['x']}
            except Exception as e:
                return {"frame_type": "invalid", "reason": f"invalid expression: {str(e)}"}
    else:
        # no '=' implies a direct x(t)=expr form
        try:
            sp_expr = sp.sympify(func_str, locals={'c': c_sym, 't': t})
            sp_expr.subs(t,0)
            v_expr = sp.diff(sp_expr, t)
            # reject non-inertial frames
            if t in v_expr.free_symbols:
                return {"frame_type": "invalid", "reason": "non-inertial frame: velocity depends on t"}
            v = float(v_expr.subs(t, 0))
            if abs(v) >= 1:
                return {"frame_type": "invalid", "reason": "speed >= c not allowed"}
            return {"frame_type": "function", "variable": "x", "expr_str": func_str}
        except Exception as e:
            return {"frame_type": "invalid", "reason": f"invalid expression: {str(e)}"}

    return {"frame_type": "invalid", "reason": "unrecognized input format or variable name"}

def precompute_frames(frames, t_vals):
    # precompute the trajectory for all function frames
    for frame_name, frame_data in frames.items():
        if frame_data["frame_type"] == "function":
            try:
                frame_data["precomputed"] = evaluate_motion_function(frame_data["expr_str"], t_vals)
            except Exception:
                frame_data["precomputed"] = None
        else:
            frame_data["precomputed"] = None

def lorentz_transform(t_arr, x_arr, v):
    # lorentz transformation using velocity v
    gamma = 1.0 / np.sqrt(1 - v**2)
    t_prime = gamma * (t_arr - v * x_arr)
    x_prime = gamma * (x_arr - v * t_arr)
    return t_prime, x_prime

def get_reference_frame_velocity_and_gamma(frames, reference_frame):
    # compute absolute velocity of reference_frame including relativistic composition
    if reference_frame not in frames:
        return 0.0, 1.0
    
    frame_data = frames[reference_frame]
    frame_type = frame_data.get("frame_type", None)
    creation_ref = frame_data.get("creation_ref_frame", "Static Frame")

    # static frame velocity is zero
    if reference_frame == "Static Frame":
        return 0.0, 1.0

    if frame_type == "function" and frame_data["precomputed"] is not None:
        # find local velocity from x(t) definition
        t = sp.Symbol('t', real=True)
        expr = sp.sympify(frame_data["expr_str"], locals={'c': c_sym, 't': t}).subs(c_sym, c)
        v_expr = sp.diff(expr, t)
        v_local = float(v_expr.subs(t, 0))

        # retrieve parent frame velocity
        v_parent, gamma_parent = get_reference_frame_velocity_and_gamma(frames, creation_ref)

        # apply formula for relativistic velocity addition
        v_abs = (v_parent + v_local) / (1 + v_parent * v_local)
        gamma = 1.0 / np.sqrt(1 - v_abs**2) if abs(v_abs) < 1 else np.inf
        return v_abs, gamma

    elif frame_type == "event":
        # event frames do not have a defined velocity
        return get_reference_frame_velocity_and_gamma(frames, creation_ref)

    else:
        # default to parent frame velocity
        return get_reference_frame_velocity_and_gamma(frames, creation_ref)

def relative_velocity(v_target, v_source):
    # relative velocity formula
    return (v_target - v_source) / (1 - v_target * v_source)

def create_plot(frames, t_prime_value, reference_frame):
    # construct plot of transformations at chosen t'
    v_ref, gamma_ref = get_reference_frame_velocity_and_gamma(frames, reference_frame)
    t_value = t_prime_value / gamma_ref

    fig = go.Figure()
    t_index = np.abs(time_steps - t_value).argmin()
    colors = ["blue", "red", "green", "purple", "orange"]

    # set event markers and colors
    event_colors = ["black", "red", "blue", "green", "purple"]
    event_symbols = ["x", "star", "diamond", "triangle-up", "circle-open"]
    event_index = 0  # increment event index

    y_min, y_max = 0, 0
    function_frames = []
    color_i = 0

    frames_data = {}

    for frame_name, frame_data in frames.items():
        if not frame_data.get("display", False):
            continue

        creation_ref = frame_data.get("creation_ref_frame", "Static Frame")
        v_creation, gamma_creation = get_reference_frame_velocity_and_gamma(frames, creation_ref)

        # compute relative velocity between creation and current reference frame
        v_rel = relative_velocity(v_ref, v_creation)

        if frame_data["frame_type"] == "event":
            # transform event coordinates from creation frame
            t_val = frame_data["t_value"]
            x_val = frame_data["x_value"]
            t_arr = np.array([t_val])
            x_arr = np.array([x_val])
            t_prime_event, x_prime_event = lorentz_transform(t_arr, x_arr, v_rel)
            t_mark = t_prime_event[0]
            x_mark = x_prime_event[0]
            frames_data[frame_name] = {
                "frame_type": "event",
                "t_value": float(t_val),
                "x_value": float(x_val),
                "t_prime": float(t_mark),
                "x_prime": float(x_mark),
                "creation_ref_frame": creation_ref
            }

            # select marker for event
            event_color = event_colors[event_index % len(event_colors)]
            event_symbol = event_symbols[event_index % len(event_symbols)]
            event_index += 1

            fig.add_trace(go.Scattergl(
                x=[t_mark],
                y=[x_mark],
                mode="markers",
                marker=dict(color=event_color, size=10, symbol=event_symbol),
                name=f"{frame_name} (event)"
            ))

            y_min = min(y_min, x_mark)
            y_max = max(y_max, x_mark)

        elif frame_data["frame_type"] == "function" and frame_data["precomputed"] is not None:
            # transform function trajectory from creation frame
            x = frame_data["precomputed"]
            t = time_steps
            t_prime, x_prime = lorentz_transform(t, x, v_rel)

            sort_idx = np.argsort(t_prime)
            t_prime_sorted = t_prime[sort_idx]
            x_prime_sorted = x_prime[sort_idx]

            t_mark = t_prime[t_index]
            x_mark = x_prime[t_index]

            y_min = min(y_min, np.min(x_prime))
            y_max = max(y_max, np.max(x_prime))

            function_frames.append((frame_name, frame_data["variable"], frame_data["expr_str"], t_mark, x_mark))

            fig.add_trace(go.Scattergl(
                x=t_prime_sorted,
                y=x_prime_sorted,
                mode="lines",
                line=dict(color=colors[color_i % len(colors)], dash="dash"),
                name=f"{frame_name} (function)"
            ))
            fig.add_trace(go.Scattergl(
                x=[t_mark],
                y=[x_mark],
                mode="markers",
                marker=dict(color=colors[color_i % len(colors)], size=10),
                name=f"{frame_name} Position"
            ))
            color_i += 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    y_margin = 0.1 * (y_max - y_min) if y_max > y_min else 1.0
    final_y_min = y_min - y_margin
    final_y_max = y_max + y_margin

    # record transformed data for each function frame
    for (frame_name, variable, expr_str, t_mark, x_mark) in function_frames:
        frames_data[frame_name] = {
            "frame_type": "function",
            "variable": variable,
            "expr_str": expr_str,
            "t_prime": float(t_mark),
            "x_prime": float(x_mark),
            "creation_ref_frame": frames[frame_name].get("creation_ref_frame", "Unknown")
        }

    fig.update_yaxes(range=[final_y_min, final_y_max])
    fig.update_layout(
        autosize=True,
        title=f"lorentz-transformed motion at t' = {t_prime_value:.2f}s (ref: {reference_frame})",
        xaxis=dict(title="transformed time (t')", gridcolor="lightgray"),
        yaxis=dict(title="transformed position (x')", gridcolor="lightgray"),
        legend_title="frames",
        paper_bgcolor="white",
        plot_bgcolor="whitesmoke",
    )

    return fig, json.dumps(frames_data)

precompute_frames(frames, time_steps)

app.layout = html.Div([
    dash_split_pane.DashSplitPane(
        id="split-pane",
        children=[
            html.Div(
                style={"padding": "20px", "overflow": "auto", "height": "100%"},
                children=[
                    html.H1("1D special relativity visualizer", className="text-center my-3"),
                    html.Hr(),
                    dbc.Button(
                        "show/hide parameters",
                        id="toggle-params",
                        color="secondary",
                        className="mb-2"
                    ),

                    dbc.Collapse(
                        id="collapse-params",
                        is_open=False,
                        children=[
                            html.Label("set parameters:", className="mt-3"),
                            dbc.InputGroup([
                                dbc.InputGroupText("time interval (t_max):"),
                                dbc.Input(id="t-max-input", type="number", value=100, step=1),
                            ], className="mb-2"),
                            dbc.InputGroup([
                                dbc.InputGroupText("resolution:"),
                                dbc.Input(id="resolution-input", type="number", value=100, step=50),
                            ], className="mb-2"),
                            dbc.InputGroup([
                                dbc.InputGroupText("play speed (multiplier):"),
                                dbc.Input(id="play-speed-input", type="number", value=1.0, step=0.1),
                            ], className="mb-2"),
                            dbc.Button("apply", id="apply-button", color="success", className="mb-3"),
                        ]
                    ),
                    html.Br(),
                    html.Label("add/edit frame/event:", className="mt-3"),
                    dbc.Input(
                        id="frame-name", type="text", placeholder="frame/event name (e.g., frame 2)",
                        className="mb-2"
                    ),
                    dbc.Input(
                        id="motion-function", type="text", 
                        placeholder="motion input (e.g., `x=50` or `x=0.5*t` or `t=5, x=5`)",
                        className="mb-2"
                    ),
                    dbc.Button("add/update frame", id="add-frame", color="primary", className="mb-3 me-2"),
                    dbc.Button("delete frame", id="delete-frame", color="danger", className="mb-3"),
                    html.Div(id="frame-feedback", className="text-muted"),

                    html.Label("select frames to display:", className="mt-3"),
                    dbc.Checklist(
                        id="frame-selector",
                        options=[{"label": name, "value": name} for name in frames],
                        value=[name for name in frames if frames[name]["display"]],
                        inline=True,
                    ),
                    html.Div(id="frame-formulas-display", className="mt-3"),
                    html.Label("select reference frame:", className="mt-3"),
                    dcc.Dropdown(
                        id="reference-frame",
                        options=[{"label": name, "value": name} for name in frames],
                        value="Static Frame",
                        clearable=False,
                        className="mb-3"
                    ),
                    html.Label("time slider (t') [s]:", className="mt-3"),
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=t_max,
                        value=0,
                        step=t_max / default_resolution,
                        marks={float(i): f"{i:.1f}" for i in np.linspace(0, t_max, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    dbc.InputGroup([
                        dbc.InputGroupText("set time (t'):"),
                        dbc.Input(id="specific-time-input", type="number", placeholder="set t'", step=0.1),
                        dbc.Button("go", id="go-time-button", color="primary")
                    ], className="my-2"),
                    html.Div([
                        dbc.Button("play/pause", id="play-pause-button", color="primary", className="mt-3 me-2"),
                        dbc.Button("replay", id="replay-button", color="info", className="mt-3 ms-2"),
                    ]),
                    html.Hr(),
                ]
            ),
            html.Div(
                style={"width": "100%", "height": "100%", "overflow": "auto"},
                children=[
                    dcc.Graph(
                        id="motion-plot",
                        config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
                        style={"height": "100%", "width": "100%"}
                    ),
                    dcc.Interval(id="interval", interval=play_interval_ms, n_intervals=0, disabled=True),
                ]
            ),
        ],
        split="vertical",
        size=300,
        paneStyle={"height": "100%"}
    ),
    dcc.Store(id="frames-data", storage_type='memory'),
    dcc.Store(id="old-gamma-store", data=1.0),
    html.Div(id='dummy-output', style={'display':'none'})
])

@app.callback(
    Output("collapse-params", "is_open"),
    [Input("toggle-params", "n_clicks")],
    [State("collapse-params", "is_open")]
)
def toggle_collapse(n_clicks, is_open):
    # show or hide the parameter settings
    if n_clicks:
        return not is_open
    return is_open

app.clientside_callback(
    """
    function(size) {
        window.dispatchEvent(new Event('resize'));
        return "";
    }
    """,
    Output('dummy-output', 'children'),
    Input('split-pane', 'size')
)

@app.callback(
    [Output("motion-plot", "figure"), 
     Output("frame-selector", "options"), 
     Output("reference-frame", "options"),
     Output("frame-selector", "value"),
     Output("time-slider", "marks"),
     Output("time-slider", "max"),
     Output("time-slider", "step"),
     Output("frames-data", "data")],
    [Input("frame-selector", "value"), Input("time-slider", "value"),
     Input("add-frame", "n_clicks"), Input("delete-frame", "n_clicks"),
     Input("reference-frame", "value"), Input("apply-button", "n_clicks")],
    [State("frame-name", "value"), State("motion-function", "value"),
     State("t-max-input", "value"), State("resolution-input", "value"), State("play-speed-input", "value")]
)
def update_plot(selected_frames, t_prime_value, add_clicks, delete_clicks, reference_frame, apply_clicks,
                frame_name, motion_function, new_t_max, new_res, speed_multiplier):
    # respond to user adjustments of parameters and frames
    global t_max, default_resolution, time_steps, play_interval_ms, base_play_interval_ms, frames
    changed_id = ctx.triggered_id

    # update t_max, resolution, and speed
    if changed_id == "apply-button" and new_t_max and new_res and speed_multiplier:
        t_max = float(new_t_max)
        default_resolution = int(new_res)
        if speed_multiplier <= 0:
            speed_multiplier = 1.0
        play_interval_ms = base_play_interval_ms / float(speed_multiplier)
        time_steps = np.linspace(0, t_max, default_resolution)
        precompute_frames(frames, time_steps)

    # add or modify a frame or event definition
    if changed_id == "add-frame" and frame_name and motion_function:
        parsed = parse_input_function(motion_function)
        if parsed["frame_type"] != "invalid":
            frames[frame_name] = {
                "frame_type": parsed["frame_type"],
                "display": True,
                "precomputed": None,
                "creation_ref_frame": reference_frame
            }
            if parsed["frame_type"] == "function":
                frames[frame_name]["variable"] = parsed["variable"]
                frames[frame_name]["expr_str"] = parsed["expr_str"]
            elif parsed["frame_type"] == "event":
                frames[frame_name]["t_value"] = parsed["t_value"]
                frames[frame_name]["x_value"] = parsed["x_value"]
            precompute_frames(frames, time_steps)
            if frame_name not in selected_frames:
                selected_frames.append(frame_name)

    # remove a frame
    if changed_id == "delete-frame" and frame_name in frames:
        frames.pop(frame_name)
        if frame_name in selected_frames:
            selected_frames.remove(frame_name)

    for f in frames:
        frames[f]["display"] = f in selected_frames

    v_ref, gamma_ref = get_reference_frame_velocity_and_gamma(frames, reference_frame)

    t_max_prime = t_max * gamma_ref

    if t_prime_value > t_max_prime:
        t_prime_value = t_max_prime

    marks_prime = {float(gamma_ref*i): f"{(gamma_ref*i):.1f}" for i in np.linspace(0, t_max, 11)}
    slider_step = t_max_prime / default_resolution if default_resolution > 0 else 0.01

    fig, frames_data_json = create_plot(frames, t_prime_value, reference_frame)

    frame_options = [{"label": name, "value": name} for name in frames]

    return fig, frame_options, frame_options, selected_frames, marks_prime, t_max_prime, slider_step, frames_data_json

@app.callback(
    Output("frame-feedback", "children"),
    [Input("add-frame", "n_clicks"), Input("delete-frame", "n_clicks")],
    [State("frame-name", "value"), State("motion-function", "value")]
)
def add_or_update_frame_feedback(add_clicks, delete_clicks, frame_name, motion_function):
    # provide feedback on frame operations
    changed_id = ctx.triggered_id
    if changed_id == "add-frame":
        if not frame_name or not motion_function:
            return "error: frame/event name and motion function required"
        parsed = parse_input_function(motion_function)
        if parsed["frame_type"] == "invalid":
            return f"error in motion function: {parsed.get('reason', 'unknown error')}"
        return f"frame '{frame_name}' added/updated"
    elif changed_id == "delete-frame":
        if frame_name in frames:
            return f"frame '{frame_name}' deleted"
        return "error: frame not found"
    return ""

@app.callback(
    Output("frame-formulas-display", "children"),
    [Input("frame-selector", "value"),
     Input("frames-data", "data"),
     Input("reference-frame", "value")],
    prevent_initial_call=True
)
def update_frame_formulas_display(selected_frames, frames_data_json, reference_frame):
    if not frames_data_json:
        return []
    frames_data = json.loads(frames_data_json)
    t = sp.Symbol('t', real=True)
    tprime = sp.Symbol("t'", real=True)
    v_rel_sym = sp.Symbol('v_{rel}', real=True)
    gamma_rel_sym = sp.Symbol(r'\gamma_{rel}', real=True, positive=True)

    children = []

    def get_v(fr_name):
        v, gamma = get_reference_frame_velocity_and_gamma(frames, fr_name)
        return v

    v_ref = get_v(reference_frame)

    for frame in selected_frames:
        if frame not in frames_data:
            continue
        frame_info = frames_data[frame]
        frame_type = frame_info["frame_type"]
        creation_ref = frame_info.get("creation_ref_frame", "unknown")
        v_creation = get_v(creation_ref)

        numeric_v_rel = (v_ref - v_creation)/(1 - v_ref*v_creation)
        numeric_gamma_rel = 1/sp.sqrt(1 - numeric_v_rel**2)

        if frame_type == "function":
            expr_str = frame_info['expr_str']
            sp_expr = sp.sympify(expr_str, locals={'c': c_sym, 't': t}).subs(c_sym, c)
            var = frame_info['variable']
            t_val = frame_info["t_prime"]
            x_val = frame_info["x_prime"]

            # original formula in creation frame
            expr_latex = sp.latex(sp_expr)
            orig_div = html.Div([
                html.H5(frame),
                dcc.Markdown(
                    f"$${var}(t) = {expr_latex}$$ (in <u>{creation_ref}</u> frame)",
                    mathjax=True,
                    dangerously_allow_html=True
                )
            ], style={"marginBottom": "20px"})

            # apply lorentz transform
            tprime_expr_sym = gamma_rel_sym*(t - v_rel_sym*sp_expr)
            xprime_expr_sym = gamma_rel_sym*(sp_expr - v_rel_sym*t)

            try:
                sol_t = sp.solve(sp.Eq(tprime_expr_sym, tprime), t)
                if sol_t:
                    t_in_terms_of_tprime = sol_t[0]
                    xprime_in_terms_of_tprime = xprime_expr_sym.subs(t, t_in_terms_of_tprime)
                    xprime_simpl = sp.simplify(xprime_in_terms_of_tprime)
                    xprime_latex = sp.latex(xprime_simpl)
                    xprime_numeric = xprime_simpl.subs({
                        v_rel_sym: numeric_v_rel,
                        gamma_rel_sym: numeric_gamma_rel
                    })
                    xprime_numeric_simpl = sp.nsimplify(sp.simplify(xprime_numeric), [sp.sqrt(2), sp.pi, sp.E])
                    try:
                        ratio = sp.simplify(xprime_numeric_simpl / tprime)
                        ratio_val = ratio.subs(tprime, 1)
                        numeric_approx_latex = f"{float(ratio_val):.10f} * t'"
                    except Exception:
                        val_at_1 = xprime_numeric_simpl.subs(tprime, 1)
                        numeric_approx_latex = f"{float(val_at_1):.10f}"

                    trans_div = html.Div([
                        dcc.Markdown(
                            f"$${var}'(t') = {xprime_latex} = {numeric_approx_latex}$$ (in <u>{reference_frame}</u> frame)\n\n",
                            mathjax=True,
                            dangerously_allow_html=True
                        ),
                        html.P(f"{var}'({t_val:.2f}) = {x_val:.2f}")
                    ], style={"marginBottom": "20px"})

                    children.append(html.Div([orig_div, trans_div]))
                else:
                    # no closed form solution found
                    no_sol_div = html.Div([
                        orig_div,
                        html.P("no closed-form solution for $x'(t')$ found")
                    ], style={"marginBottom": "20px"})
                    children.append(no_sol_div)
            except Exception:
                # no closed form solution found
                no_sol_div = html.Div([
                    orig_div,
                    html.P("no closed-form solution for $x'(t')$ found")
                ], style={"marginBottom": "20px"})
                children.append(no_sol_div)

        elif frame_type == "event":
            # show transformed coordinates for events
            t_val = frame_info["t_value"]
            x_val = frame_info["x_value"]
            t_prime_val = frame_info["t_prime"]
            x_prime_val = frame_info["x_prime"]

            children.append(
                html.Div([
                    html.H5(frame),
                    dcc.Markdown(
                        f"$$(t, x) = ({t_val:.2f}, {x_val:.2f})$$ (in <u>{creation_ref}</u> frame)\n\n",
                        mathjax=True,
                        dangerously_allow_html=True
                    ),
                    html.P(f"(t', x') = ({t_prime_val:.2f}, {x_prime_val:.2f})")
                ], style={"marginBottom": "20px"})
            )

        else:
            # handle invalid frame type
            children.append(
                html.Div([
                    html.H5(frame),
                    html.P("invalid frame")
                ], style={"marginBottom": "20px"})
            )

    return children

@app.callback(
    [Output("time-slider", "value"),
     Output("interval", "disabled"),
     Output("play-pause-button", "children"),
     Output("old-gamma-store", "data")],
    [Input("interval", "n_intervals"),
     Input("replay-button", "n_clicks"),
     Input("go-time-button", "n_clicks"),
     Input("play-pause-button", "n_clicks"),
     Input("time-slider", "value"),
     Input("reference-frame", "value")],
    [State("specific-time-input", "value"),
     State("interval", "disabled"),
     State("old-gamma-store", "data")]
)
def update_time_slider(n_intervals, replay_clicks, go_time_clicks, play_pause_clicks, current_t_prime,
                       reference_frame, new_t_prime, interval_disabled, old_gamma):
    # control time slider and animation states
    changed_id = ctx.triggered_id
    play_pause_button_text = "play/pause"
    was_playing = not interval_disabled

    v_ref, gamma_ref = get_reference_frame_velocity_and_gamma(frames, reference_frame)
    t_max_prime = t_max * gamma_ref

    if changed_id == "reference-frame":
        # recompute time scale after frame change
        t = current_t_prime / old_gamma
        current_t_prime = t * gamma_ref
        if current_t_prime > t_max_prime:
            current_t_prime = t_max_prime
        if current_t_prime < 0:
            current_t_prime = 0
        old_gamma = gamma_ref

    if changed_id == "replay-button":
        saved_state = was_playing
        interval_disabled = True
        current_t_prime = 0
        interval_disabled = not saved_state
        return current_t_prime, interval_disabled, play_pause_button_text, old_gamma

    if changed_id == "go-time-button" and new_t_prime is not None:
        val = float(new_t_prime)
        if val < 0:
            val = 0
        if val > t_max_prime:
            val = t_max_prime
        current_t_prime = val
        return current_t_prime, interval_disabled, play_pause_button_text, old_gamma

    if changed_id == "play-pause-button":
        if play_pause_clicks is None:
            interval_disabled = True
        else:
            interval_disabled = not interval_disabled
        return current_t_prime, interval_disabled, play_pause_button_text, old_gamma

    if changed_id == "interval":
        if not interval_disabled and current_t_prime < t_max_prime:
            step = t_max_prime / default_resolution
            val = current_t_prime + step
            if val > t_max_prime:
                val = t_max_prime
            current_t_prime = val

    if current_t_prime >= t_max_prime:
        interval_disabled = True

    if changed_id != "reference-frame":
        old_gamma = gamma_ref

    return current_t_prime, interval_disabled, play_pause_button_text, old_gamma

@app.callback(
    Output("interval", "interval"),
    [Input("apply-button", "n_clicks")],
    [State("play-speed-input", "value")]
)
def update_interval_speed(apply_clicks, speed_multiplier):
    # adjust animation interval based on play speed
    if speed_multiplier and speed_multiplier > 0:
        return int(base_play_interval_ms / float(speed_multiplier))
    return base_play_interval_ms

if __name__ == "__main__":
    app.run_server(debug=True)
