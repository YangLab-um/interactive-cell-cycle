import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.integrate import solve_ivp
from utils.ode_models import Guan2008Model
from utils.model_interface import create_parameter_sliders

# Initial model and plot setup
model = Guan2008Model()

df_ts = pd.DataFrame(data={'Time (min)': np.zeros(2), 'Total CyclinB1': np.zeros(2), 'Active CyclinB1:Cdk1 complex': np.zeros(2)})
fig_time_series = px.line(df_ts, x='Time (min)', y=['Total CyclinB1', 'Active CyclinB1:Cdk1 complex'], labels={'value': 'Concentration (nM)'})
fig_time_series.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'),
                              margin=dict(l=0, r=0, t=0, b=0, pad=0), hovermode=False)

df_pp = pd.DataFrame(data={'Total CyclinB1 (nM)': np.zeros(2), 'B nullcline': np.zeros(2), 'C nullcline': np.zeros(2), 'Trajectory': np.zeros(2)})
fig_phase_plane = px.line(df_pp, x='Total CyclinB1 (nM)', y=['B nullcline', 'C nullcline', 'Trajectory'], labels={'value': 'Active CyclinB1:Cdk1 complex (nM)'},
                          line_dash='variable', line_dash_map={'B nullcline': 'solid', 'C nullcline': 'solid', 'Trajectory': 'dot'},
                          color_discrete_sequence=['#636EFA', '#EF553B', 'rgba(0,0,0,0.25)'],)
fig_phase_plane.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'),
                              margin=dict(l=0, r=0, t=0, b=0, pad=0), hovermode=False)

df_rr = pd.DataFrame(data={'Active CyclinB1:Cdk1 complex (nM)': np.zeros(2), 'Degradation' : np.zeros(2), 'Cdc25' : np.zeros(2), 'Wee1' : np.zeros(2)})
fig_reaction_rates = px.line(df_rr, x='Active CyclinB1:Cdk1 complex (nM)', y=['Degradation', 'Cdc25', 'Wee1'], labels={'value': 'Reaction rate (nM/min)'})
fig_reaction_rates.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'),
                                 margin=dict(l=0, r=0, t=0, b=0, pad=0), hovermode=False)

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.LUX]
)

app.title = 'Interactive Cell Cycle Mathematical Models'
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dcc.Markdown('# Interactive Cell Cycle Model - [Yang Lab](http://www-personal.umich.edu/~qiongy/)', className='text-center text-primary mt-4'),
                className='text-center text-primary',
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Hr(className='border border-primary'),
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(model.introduction, mathjax=True),
                className='mt-2'
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Hr(className='border border-primary mt-0 mb-4'),
            ),
        ),
        dbc.Row(
            [
                dbc.Col([
                        dbc.Row([
                                dbc.Col(
                                    html.H3('Parameters'),
                                    width=8,
                                ),
                                dbc.Col(
                                    dbc.Button('Reset values', id='reset-button-parameters', color='primary', className='py-1'),
                                    width=4, className='d-flex justify-content-end',
                                ),
                        ], className='mb-3 g-0 ms-2 d-flex justify-content-between align-items-center'),
                    ] + create_parameter_sliders(model.slider_information) + [
                        dbc.Row([
                                dbc.Col(
                                    html.H3('Initial condition'),
                                ),
                                dbc.Col(
                                    dbc.Button('Reset values', id='reset-button-initial-conditions', color='primary', className='py-1'),
                                    width=4, className='d-flex justify-content-end',
                                ),
                        ], className='mb-3 mt-4 g-0 ms-2 d-flex justify-content-between align-items-center'),
                        dbc.Row([
                                dbc.Col(
                                    dcc.Markdown('Total Cyclin B1', style={"padding-left": "7.5%"}), 
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Slider(id='cyclin-b1-initial-condition-slider', 
                                               min=0.000001, max=120.0, step=0.1, value=75.0, 
                                               marks=None, tooltip={'placement': 'bottom', 'always_visible': True},),
                                ),
                        ]),
                        dbc.Row([
                                dbc.Col(
                                    dcc.Markdown('Active Cyclin B1:Cdk1', style={"padding-left": "7.5%"}), 
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Slider(id='cdk1-initial-condition-slider', min=0.000001, max=120.0, step=0.1, value=35.0, 
                                               marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
                                ),
                        ]),
                         dbc.Row([
                                dbc.Col(
                                    html.H3('Simulation time'),
                                ),
                                dbc.Col(
                                    dbc.Button('Reset value', id='reset-button-simulation-time', color='primary', className='py-1'),
                                    width=4, className='d-flex justify-content-end',
                                ),
                        ], className='mb-3 mt-4 g-0 ms-2 d-flex justify-content-between align-items-center'),
                        dbc.Row([
                                dbc.Col(
                                    dcc.Markdown('Time (min)', style={"padding-left": "7.5%"}), 
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Slider(id='simulation-time-slider', 
                                               min=1, max=6000.0, step=1, value=2000.0, 
                                               marks=None, tooltip={'placement': 'bottom', 'always_visible': True},),
                                ),
                        ]),
                ]),
                dbc.Col(
                    [
                        dbc.Row(
                            dcc.Loading(
                                children=[dcc.Graph(id='time-series', figure=fig_time_series)],
                                parent_className='loading_wrapper',
                                type='dot',
                                color='#3CB371',
                            ),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(
                                        children=[dcc.Graph(id='phase-plane', figure=fig_phase_plane)],
                                        parent_className='loading_wrapper',
                                        type='dot',
                                        color='#3CB371',
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Loading(
                                        children=[dcc.Graph(id='reaction-rates', figure=fig_reaction_rates)],
                                        parent_className='loading_wrapper',
                                        type='dot',
                                        color='#3CB371',
                                    ),
                                    width=6,
                                ),
                            ]
                        ),
                    ]
                ),
            ],
        ),
        dbc.Row(
            dbc.Col(
                html.Hr(className='border border-primary mt-4 mb-3'),
            ),
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Markdown('''Source code: [Github](https://github.com/YangLab-um/interactive-cell-cycle)''', mathjax=True),
                ),
            ],
        ),

    ], fluid=True
)

@app.callback(
    [Output('time-series', 'figure'), Output('phase-plane', 'figure'), Output('reaction-rates', 'figure')],
    Input('cyclin-b1-initial-condition-slider', 'value'), Input('cdk1-initial-condition-slider', 'value'),
    Input('simulation-time-slider', 'value'),
    State('time-series', 'figure'), State('phase-plane', 'figure'), State('reaction-rates', 'figure'),
    [Input(f'{parameter}-slider', 'value') for parameter in model.slider_information.keys()],)
def slider_callback(B0, C0, sim_time, fig_time_series, fig_phase_plane, fig_reaction_rates, *parameters):
    parameters = [float(parameter) for parameter in parameters]
    parameter_dict = dict(zip(model.slider_information.keys(), parameters))
    model.set_parameters(parameter_dict)
    model.set_initial_conditions([float(B0), float(C0)])
    solution = solve_ivp(model.equations, [0, float(sim_time)], model.initial_conditions, 
                        args=(model.parameters,))

    # Time series plot
    fig_time_series['data'][0]['x'] = solution.t
    fig_time_series['data'][0]['y'] = solution.y[0,:]
    fig_time_series['data'][1]['x'] = solution.t
    fig_time_series['data'][1]['y'] = solution.y[1,:]

    # Phase plane plot
    C_values = np.linspace(0, 120, 100)
    B_nullcline, C_nullcline = model.nullclines(C_values, model.parameters)
    fig_phase_plane['data'][0]['x'] = B_nullcline
    fig_phase_plane['data'][0]['y'] = C_values
    fig_phase_plane['data'][1]['x'] = C_nullcline
    fig_phase_plane['data'][1]['y'] = C_values
    fig_phase_plane['data'][2]['x'] = solution.y[0,:]
    fig_phase_plane['data'][2]['y'] = solution.y[1,:]

    # Reaction rates plot
    HD, HT, HW = model.interaction_terms(C_values, model.parameters)
    fig_reaction_rates['data'][0]['x'] = C_values
    fig_reaction_rates['data'][0]['y'] = HD
    fig_reaction_rates['data'][1]['x'] = C_values
    fig_reaction_rates['data'][1]['y'] = HT
    fig_reaction_rates['data'][2]['x'] = C_values
    fig_reaction_rates['data'][2]['y'] = HW

    return fig_time_series, fig_phase_plane, fig_reaction_rates


@app.callback(
    [Output(f'{parameter}-slider', 'value') for parameter in model.slider_information.keys()],
    Input('reset-button-parameters', 'n_clicks'),
)
def reset_parameters(n_clicks):
    default_model = Guan2008Model()
    default_parameters = [default_model.slider_information[parameter]['initial_value'] for parameter in default_model.slider_information.keys()]
    return default_parameters


@app.callback(
    [Output('cyclin-b1-initial-condition-slider', 'value'), Output('cdk1-initial-condition-slider', 'value')],
    Input('reset-button-initial-conditions', 'n_clicks'),
)
def reset_initial_conditions(n_clicks):
    default_model = Guan2008Model()
    return default_model.initial_conditions[0], default_model.initial_conditions[1]

@app.callback(
    Output('simulation-time-slider', 'value'),
    Input('reset-button-simulation-time', 'n_clicks'),
)
def reset_simulation_time(n_clicks):
    default_model = Guan2008Model()
    return default_model.simulation_time


if __name__ == '__main__':
    app.run_server(debug=True)