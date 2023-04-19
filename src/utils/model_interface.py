from dash import dcc, html
import dash_bootstrap_components as dbc


def create_parameter_sliders(slider_info_dict):
    """
    Generate a list of rows containing a number display and slider for each parameter

    Parameters
    ----------
    slider_info_dict : dict
        Nested dictionary with parameter names as keys and a dict for value.
        The value-dict has keys determining the initial value, min, max, step, description, and units.
        For example:
        slider_info_dict = {
            'kS': {
                'initial_value': 0.0, 
                'description': 'Protein synthesis rate', 
                'min': 0.0, 
                'max': 4.0, 
                'step': 0.01, 
                'units': '1/min'
            },
        }

    Returns
    -------
    list
        A list of dbc.Row() elements. Each row contains a display that shows
        the current parameter value and a slider. The ID of the display is
        '<param_name>-display'. Similarly, the ID of the slider is
        '<param_name>-slider'.
    """

    all_sliders = []
    for name in slider_info_dict.keys():
        init_val = slider_info_dict[name]['initial_value']
        min_val = slider_info_dict[name]['min']
        max_val = slider_info_dict[name]['max']
        step = slider_info_dict[name]['step']
        units = slider_info_dict[name]['units']
        display = dcc.Markdown(f"{name} ({units})", id=f"{name}-display",
                          style={"padding-left": "10%"})
        tooltip = dbc.Tooltip(slider_info_dict[name]['description'], target=f"{name}-display",
                              placement="bottom", style={"font-size": "0.8rem"})
        slider = dcc.Slider(id=f'{name}-slider', min=min_val, max=max_val,
                            step=step, value=init_val, marks=None,
                            tooltip={"placement": "bottom", "always_visible": True})
        row = dbc.Row([ dbc.Col(display, width=3), tooltip, dbc.Col(slider, width=9), ], style={"width": "100%"})
        all_sliders.append(row)
    return all_sliders