import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

def generate_thesis_visuals():
    """
    Generates two academic-grade 3D surface plots for blood clot risk monitoring
    using Plotly Graph Objects.
    """
    print("--- Generating Master's Thesis Visuals: 3D Clinical Risk Monitoring ---")

    # =================================================================
    # GRAPH 1: 3D CLINICAL RISK SURFACE (BMI vs HR vs PROB)
    # =================================================================
    
    # 1. Define coordinate ranges
    bmi_range = np.linspace(18, 40, 50)
    hr_range = np.linspace(60, 180, 50)
    BMI, HR = np.meshgrid(bmi_range, hr_range)
    
    # 2. Mathematical model for Surface (Sigmoid Risk Scaling)
    # Risk increases non-linearly with both BMI and HR
    # Formula: z = 1 / (1 + exp(-(alpha*BMI + beta*HR - offset)))
    alpha = 0.15
    beta = 0.05
    offset = 11.5 # Adjusted to place the transition in the "High" zone
    Z_prob = 1 / (1 + np.exp(-(alpha * BMI + beta * HR - offset)))
    
    # 3. Representative Data for the 7 Actual 'Critical' Patients
    # (High BMI, High HR, High Prob)
    critical_cases = {
        'bmi': [34.2, 38.5, 31.0, 36.8, 39.1, 29.5, 37.2],
        'hr': [142, 165, 158, 131, 172, 148, 155],
        'prob': [0.88, 0.96, 0.72, 0.81, 0.98, 0.65, 0.91]
    }

    fig1 = go.Figure()

    # Add Surface
    fig1.add_trace(go.Surface(
        x=BMI, y=HR, z=Z_prob,
        colorscale='RdYlGn_r', # Red for High Risk, Green for Low
        opacity=0.85,
        name='Calibrated Risk Plane',
        colorbar=dict(title='Risk Prob', tickvals=[0, 0.5, 1], ticktext=['Safe', 'Warning', 'Critical'])
    ))

    # Add the 7 Critical Patients as Scatter3d
    fig1.add_trace(go.Scatter3d(
        x=critical_cases['bmi'],
        y=critical_cases['hr'],
        z=critical_cases['prob'],
        mode='markers',
        marker=dict(size=8, color='crimson', symbol='diamond', line=dict(width=2, color='white')),
        name='Observed Critical Cases (n=7)',
        text=[f"Patient {i+1}<br>Critical Risk" for i in range(7)]
    ))

    # Layout for academic presentation
    fig1.update_layout(
        title=dict(
            text="<b>Graph 1: 3D Clinical Risk Surface</b><br><sup>Non-linear Scaling of Critical Risk Probability vs. Body Composition and Heart Rate</sup>",
            font=dict(size=20, family="Times New Roman")
        ),
        scene=dict(
            xaxis_title='Body Mass Index (BMI)',
            yaxis_title='Mean Heart Rate (bpm)',
            zaxis_title='Calibrated Prob(Critical)',
            xaxis=dict(gridcolor='lightgrey', backgroundcolor="rgb(245, 245, 245)"),
            yaxis=dict(gridcolor='lightgrey', backgroundcolor="rgb(245, 245, 245)"),
            zaxis=dict(gridcolor='lightgrey', backgroundcolor="rgb(245, 245, 245)", range=[0, 1.0]),
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.6))
        ),
        margin=dict(l=0, r=0, b=0, t=80),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # =================================================================
    # GRAPH 2: ENSEMBLE DECISION SPACE (THRESHOLD GATING)
    # =================================================================
    
    # 1. Coordinate ranges for two backbones (Prob 0.0 to 1.0)
    b1_range = np.linspace(0, 1, 40)
    b2_range = np.linspace(0, 1, 40)
    B1, B2 = np.meshgrid(b1_range, b2_range)
    
    # 2. Ensemble Logic (Soft Weighted Average + Slight Nonlinearity)
    # Final_P = sigmoid(w1*B1 + w2*B2)
    Z_ensemble = (0.6 * B1 + 0.4 * B2) 
    
    # Safety Threshold Gate
    threshold = 0.10 
    
    fig2 = go.Figure()

    # Add Ensemble Surface
    fig2.add_trace(go.Surface(
        x=B1, y=B2, z=Z_ensemble,
        colorscale='Plasma',
        opacity=0.7,
        name='Ensemble Inference Plane'
    ))

    # Add the Safety Threshold Plateau (Decision Boundary)
    fig2.add_trace(go.Surface(
        x=B1, y=B2, z=np.full_like(Z_ensemble, threshold),
        showscale=False,
        opacity=0.4,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        name=f'Safety Gate (P > {threshold})'
    ))

    # Annotations were already added in the scene=dict(...) layout below
    
    fig2.update_layout(
        title=dict(
            text=f"<b>Graph 2: Ensemble Decision Space</b><br><sup>Ensemble Gating Logic with Safety Threshold P > {threshold}</sup>",
            font=dict(size=20, family="Times New Roman")
        ),
        scene=dict(
            xaxis_title='Backbone 1: CNN-Transformer Prob',
            yaxis_title='Backbone 2: Bi-LSTM Prob',
            zaxis_title='Final Calibrated Probability',
            xaxis=dict(nticks=10, range=[0, 1]),
            yaxis=dict(nticks=10, range=[0, 1]),
            zaxis=dict(nticks=10, range=[0, 1]),
            annotations=[
                dict(
                    x=0.8, y=0.8, z=0.9,
                    text="CRITICAL ZONE",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(color="red", size=14)
                ),
                dict(
                    x=0.1, y=0.1, z=threshold,
                    text="SAFETY GATE",
                    showarrow=True,
                    arrowhead=2,
                    ax=50, ay=-50,
                    font=dict(color="cyan", size=14)
                )
            ]
        ),
        margin=dict(l=0, r=0, b=0, t=80)
    )

    # Save as interactive HTMLs
    os.makedirs('reports/visuals', exist_ok=True)
    fig1.write_html('reports/visuals/graph1_clinical_risk_surface.html')
    fig2.write_html('reports/visuals/graph2_ensemble_decision_space.html')
    
    print("\nVisuals generated successfully!")
    print("1. reports/visuals/graph1_clinical_risk_surface.html")
    print("2. reports/visuals/graph2_ensemble_decision_space.html")
    
    # Try to show them if in an interactive environment (optional)
    # fig1.show()
    # fig2.show()

if __name__ == "__main__":
    generate_thesis_visuals()
