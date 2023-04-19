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

# Initial setup
model = Guan2008Model()

df_ts = pd.DataFrame(data={'Time (min)': np.zeros(2), 'Total CyclinB1': np.zeros(2), 'Active CyclinB1:Cdk1 complex': np.zeros(2)})
fig_time_series = px.line(df_ts, x='Time (min)', y=['Total CyclinB1', 'Active CyclinB1:Cdk1 complex'], labels={'value': 'Concentration (nM)'})
fig_time_series.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))

# TODO: Add trajectory to phase plane
df_pp = pd.DataFrame(data={'Total CyclinB1 (nM)': np.zeros(2), 'B_nullcline': np.zeros(2), 'C_nullcline': np.zeros(2)})
fig_phase_plane = px.line(df_pp, x='Total CyclinB1 (nM)', y=['B_nullcline', 'C_nullcline'], labels={'value': 'Active CyclinB1:Cdk1 complex (nM)'})
fig_phase_plane.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))

df_rr = pd.DataFrame(data={'Active CyclinB1:Cdk1 complex (nM)': np.zeros(2), 'Degradation' : np.zeros(2), 'Cdc25' : np.zeros(2), 'Wee1' : np.zeros(2)})
fig_reaction_rates = px.line(df_rr, x='Active CyclinB1:Cdk1 complex (nM)', y=['Degradation', 'Cdc25', 'Wee1'], labels={'value': 'Reaction rate (nM/min)'})
fig_reaction_rates.update_layout(legend=dict(title='', yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))

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
                html.H1('Interactive Cell Cycle Model - Yang Lab'),
                className='text-center text-primary my-4',
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Hr(className='border border-primary'),
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown(model.introduction),
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown('The model is described by the following system of ordinary differential equations:'),
            ),
        ),
        # TODO: Replace picture equations with latex equations
        dbc.Row(
            dbc.Col(
                html.Img(src=dash.get_asset_url('equations/Guan2008/cyclin_B_equation.png')),
            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Img(src=dash.get_asset_url('equations/Guan2008/cdk1_equation.png')),
            ),
        ),
        dbc.Row(
            dbc.Col(
                dcc.Markdown('''where B represents the total amount of CyclinB1 and C the concentration of active cyclinB1:Cdk1 complex
                                Additionally, subscript D represents the degradation reactions, T Cdc25 interactions, and W Wee1 interactions.''')

            ),
        ),
        dbc.Row(
            dbc.Col(
                html.Hr(className='border border-primary'),
            ),
        ),
        # Parameter sliders and plots
        dbc.Row(
            [
                # Parameter sliders
                dbc.Col([
                        dbc.Row([
                                dbc.Col(
                                    html.H3('Parameters'),
                                ),
                                dbc.Col(
                                    dbc.Button('Reset values', id='reset-button-parameters', color='primary', className='mr-1'),
                                ),
                        ]),
                    # TODO: Add tooltips to sliders with their description
                    ] + create_parameter_sliders(model.slider_information) + [
                        dbc.Row([
                                dbc.Col(
                                    html.H3('Initial conditions'),
                                ),
                                dbc.Col(
                                    dbc.Button('Reset values', id='reset-button-initial-conditions', color='primary', className='mr-1'),
                                ),
                        ]),
                        dbc.Row([
                                dbc.Col(
                                    dcc.Markdown('Total Cyclin B1'), 
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Slider(id='cyclin-b1-initial-condition-slider', 
                                               min=0.000001, max=120.0, step=0.1, value=75.0, 
                                               marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
                                ),
                        ]),
                        dbc.Row([
                                dbc.Col(
                                    dcc.Markdown('Active Cyclin B1:Cdk1 complex'), 
                                    width=4,
                                ),
                                dbc.Col(
                                    dcc.Slider(id='cdk1-initial-condition-slider', min=0.000001, max=120.0, step=0.1, value=35.0, 
                                               marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
                                ),
                        ]),
                ]),
                # Plots
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

    ], fluid=True
)

@app.callback(
    [Output('time-series', 'figure'), Output('phase-plane', 'figure'), Output('reaction-rates', 'figure')],
    Input('cyclin-b1-initial-condition-slider', 'value'), Input('cdk1-initial-condition-slider', 'value'),
    State('time-series', 'figure'), State('phase-plane', 'figure'), State('reaction-rates', 'figure'),
    [Input(f'{parameter}-slider', 'value') for parameter in model.slider_information.keys()],)
def slider_callback(B0, C0, fig_time_series, fig_phase_plane, fig_reaction_rates, *parameters):
    parameters = [float(parameter) for parameter in parameters]
    parameter_dict = dict(zip(model.slider_information.keys(), parameters))
    model.set_parameters(parameter_dict)
    model.set_initial_conditions([float(B0), float(C0)])
    solution = solve_ivp(model.equations, [0, 2000], model.initial_conditions, 
                        args=(model.parameters,), method='LSODA')

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



if __name__ == '__main__':
    app.run_server(debug=True)
