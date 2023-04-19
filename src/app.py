import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import plotly.express as px
from scipy.integrate import solve_ivp
from utils.ode_models import Parameters, Guan2008Model
from utils.model_interface import create_parameter_sliders

# Initial setup
model = Guan2008Model()
fig_time_series = px.scatter(labels={'x': ')', 'y': ''})
fig_phase_plane = px.scatter(labels={'x': '', 'y': ''})
fig_reaction_rates = px.scatter(labels={'x': '', 'y': ''})

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
                            dcc.Graph(id='time-series', figure=fig_time_series),
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id='phase-plane', figure=fig_phase_plane),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Graph(id='reaction-rates', figure=fig_reaction_rates),
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
    [Input(f'{parameter}-slider', 'value') for parameter in model.slider_information.keys()],)
def slider_callback(B0, C0, *parameters):
    parameters = [float(parameter) for parameter in parameters]
    parameter_dict = dict(zip(model.slider_information.keys(), parameters))
    model.set_parameters(parameter_dict)
    model.set_initial_conditions([float(B0), float(C0)])

    # Time series plot
    solution = solve_ivp(model.equations, [0, 2000], model.initial_conditions, 
                        args=(model.parameters,), method='LSODA')
    fig_time_series = px.scatter(labels={'x': '', 'y': ''})
    fig_time_series.add_scatter(x=solution.t, y=solution.y[0,:], mode='lines', name='Total CyclinB1', line_color='#1f77b4')
    fig_time_series.add_scatter(x=solution.t, y=solution.y[1,:], mode='lines', name='Active CyclinB1:Cdk1 complex', line_color='#d62728')
    fig_time_series.update_layout(legend=dict(yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))
    fig_time_series.update_xaxes(title_text='Time (min)')
    fig_time_series.update_yaxes(title_text='Concentration (nM)')

    C_values = np.linspace(0, 120, 100)
    # Phase plane plot
    B_nullcline, C_nullcline = model.nullclines(C_values, model.parameters)
    fig_phase_plane = px.scatter(labels={'x': '', 'y': ''})
    fig_phase_plane.add_scatter(x=B_nullcline, y=C_values, mode='lines', name='Horizontal nullcline', line_color='#1f77b4')
    fig_phase_plane.add_scatter(x=C_nullcline, y=C_values, mode='lines', name='Vertical nullcline', line_color='#d62728')
    fig_phase_plane.add_scatter(x=solution.y[0,:], y=solution.y[1,:], mode='lines', name='Trajectory', line_color='#7f7f7f', line_dash='dash', opacity=0.5)
    fig_phase_plane.update_layout(legend=dict(yanchor='top', y=1.2, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))
    fig_phase_plane.update_xaxes(title_text='Total Cyclin B1 (nM)' , range=[15, 110])
    fig_phase_plane.update_yaxes(title_text='Active CyclinB1:Cdk1 (nM)', range=[0, 100])

    # Reaction rates plot
    HD, HT, HW = model.interaction_terms(C_values, model.parameters)
    fig_reaction_rates = px.scatter(labels={'x': '', 'y': ''})
    fig_reaction_rates.add_scatter(x=C_values, y=HD, mode='lines', name='Degradation', line_color='#1f77b4')
    fig_reaction_rates.add_scatter(x=C_values, y=HT, mode='lines', name='Cdc25', line_color='#d62728')
    fig_reaction_rates.add_scatter(x=C_values, y=HW, mode='lines', name='Wee1', line_color='#2ca02c')
    fig_reaction_rates.update_layout(legend=dict(yanchor='top', y=1.1, xanchor='left', x=0.0, orientation='h', bgcolor='rgba(0,0,0,0)'))
    fig_reaction_rates.update_xaxes(title_text='Active CyclinB1:Cdk1 (nM)')
    fig_reaction_rates.update_yaxes(title_text='Reaction rate (nM/min)')

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
