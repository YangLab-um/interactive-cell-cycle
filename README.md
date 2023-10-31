# Interactive Cell Cycle Model
This repository contains a web application that allows users to interact with a differential equation model of the cell cycle. The application allows to change parameters and initial conditions and observe changes to cell cycle dynamics in real time. The model was proposed in an [eLife article by Guan et al. in 2018](https://elifesciences.org/articles/33549) and aims to simulate the behavior of the cell cycle and how it is regulated by various proteins and signaling pathways.

You can try it out online [here](https://interactive-cell-cycle.onrender.com/)! For a more responsive simulation, visit the Observable version of this app [here](https://observablehq.com/d/3ad2343ac1f69afb).

The application is built using Plotly's [Dash framework](https://plotly.com/dash/). We encourage users to explore the code, modify the model, and contribute to the development of the app.

We hope that this repository will be a useful resource for those interested in learning more about the cell cycle and how it is modeled using differential equations. Don't hesitate to reach out with any questions or feedback!

## Running the application locally
To run the application locally, you will need to install [Python](https://www.python.org/downloads/) and [Git](https://git-scm.com/downloads). Once you have these installed, you can clone the repository and install the required packages using the following commands in a terminal:
```bash
    git clone
    cd interactive-cell-cycle
    pip install -r requirements.txt
```
Once the packages are installed, you can run the application using the following command:
```bash
    python app.py
```
The application should now be running on your local machine. You can access it by opening a web browser and navigating to `http://localhost:8050`.
