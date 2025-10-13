import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import logging
from dotenv import load_dotenv
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.graph.workflow import ControlLoopWorkflow
from src.models.pfd_models import PFDData
from src.utils.data_loader import DataLoader

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PFD Control Loop Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #856404;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #0c5460;
        margin: 1rem 0;
    }
    .pairing-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
    }
    .agent-message {
        background-color: #e9ecef;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #6c757d;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏭 PFD Control Loop Prediction System</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    <strong>AI-Powered Control Structure Design using Chemical Engineering Principles</strong><br>
    Multi-Agent System | RGA Analysis | SVD Controllability | LangGraph Workflow
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None
if 'pfd_data' not in st.session_state:
    st.session_state.pfd_data = None
if 'gain_matrix' not in st.session_state:
    st.session_state.gain_matrix = None
if 'agent_messages' not in st.session_state:
    st.session_state.agent_messages = []
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        st.success("✅ API Key Configured")
    else:
        st.error("❌ API Key Not Found")
        st.info("Please set GOOGLE_API_KEY in .env file")
    
    # Load config
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model': {'name': 'gemini-1.5-pro', 'temperature': 0.3},
            'agents': {}
        }
    
    # Model settings
    st.markdown("#### Model Settings")
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-pro", "gemini-1.5-flash"],
        index=0,
        help="Select the Gemini model to use"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more deterministic, Higher = more creative"
    )
    
    # Sample data selection
    st.markdown("---")
    st.markdown("### 📁 Sample Data")
    data_folder = Path("data")
    sample_files = []
    
    if data_folder.exists():
        sample_files = [f for f in data_folder.glob("*.json") if f.name.startswith('sample_')]
    
    if sample_files:
        sample_options = ["Upload Custom"] + [f.name for f in sample_files]
        selected_file = st.selectbox(
            "Select Sample PFD",
            sample_options,
            help="Choose a pre-loaded sample or upload your own"
        )
    else:
        selected_file = "Upload Custom"
        st.info("ℹ️ No sample files found in data/ folder")
    
    # Control parameters
    st.markdown("---")
    st.markdown("### 🎛️ Control Parameters")
    
    with st.expander("Advanced Settings", expanded=False):
        rga_threshold = st.slider(
            "RGA Good Pairing Threshold",
            0.5, 1.0, 0.7, 0.05,
            help="RGA values above this are considered good pairings"
        )
        
        interaction_threshold = st.slider(
            "Interaction Index Threshold",
            0.1, 0.5, 0.3, 0.05,
            help="Maximum acceptable interaction between loops"
        )
    
    # Resources
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)")
    st.markdown("- [Control Theory Basics](https://en.wikipedia.org/wiki/Control_theory)")
    st.markdown("- [RGA Analysis](https://en.wikipedia.org/wiki/Relative_gain_array)")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📥 Input Data", 
    "🔄 Run Analysis", 
    "📊 Results", 
    "🤖 Agent Activity",
    "📖 Help"
])

# ==================== TAB 1: INPUT DATA ====================
with tab1:
    st.markdown('<div class="sub-header">📥 Input PFD Data</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if selected_file != "Upload Custom":
            # Load sample file
            try:
                file_path = data_folder / selected_file
                pfd_data = DataLoader.load_json(str(file_path))
                
                st.markdown('<div class="success-box">✅ Sample file loaded successfully!</div>', 
                           unsafe_allow_html=True)
                
                # Display process info
                st.markdown(f"**Process:** {pfd_data['name']}")
                st.markdown(f"**Description:** {pfd_data['description']}")
                
                # Show JSON
                with st.expander("📄 View Raw JSON Data", expanded=False):
                    st.json(pfd_data)
                
                # Validate and store
                if DataLoader.validate_pfd_data(pfd_data):
                    st.session_state.pfd_data = pfd_data
                    st.session_state.gain_matrix = np.array(pfd_data.get('gain_matrix', []))
                    
            except Exception as e:
                st.error(f"❌ Error loading sample: {str(e)}")
                logger.error(f"Sample load error: {e}", exc_info=True)
        
        else:
            # Upload custom file
            st.markdown("#### Upload Custom PFD File")
            uploaded_file = st.file_uploader(
                "Choose a JSON file",
                type=['json'],
                help="Upload a JSON file containing PFD data in the required format"
            )
            
            if uploaded_file:
                try:
                    pfd_data = json.load(uploaded_file)
                    
                    # Validate
                    if DataLoader.validate_pfd_data(pfd_data):
                        st.markdown('<div class="success-box">✅ File uploaded and validated!</div>', 
                                   unsafe_allow_html=True)
                        
                        st.markdown(f"**Process:** {pfd_data['name']}")
                        st.markdown(f"**Description:** {pfd_data['description']}")
                        
                        with st.expander("📄 View Uploaded Data"):
                            st.json(pfd_data)
                        
                        st.session_state.pfd_data = pfd_data
                        st.session_state.gain_matrix = np.array(pfd_data.get('gain_matrix', []))
                        
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON format: {str(e)}")
                except ValueError as e:
                    st.error(f"❌ Validation error: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    logger.error(f"File upload error: {e}", exc_info=True)
    
    with col2:
        st.markdown("### 📋 Data Format Guide")
        st.markdown("""
        **Required Fields:**
        - `name`: Process name
        - `description`: Process description
        - `unit_operations`: List of units
        - `controlled_variables`: CVs with properties
        - `manipulated_variables`: MVs with properties
        - `gain_matrix`: Steady-state gain matrix
        
        **Optional:**
        - `disturbance_variables`: Disturbances
        - `time_constants`: Dynamic info
        """)
        
        if st.button("📥 Download JSON Template"):
            template = {
                "name": "Sample Process",
                "description": "Description of the process",
                "unit_operations": [
                    {"name": "R-101", "type": "reactor", "description": "Main reactor"}
                ],
                "controlled_variables": [
                    {
                        "name": "T_reactor",
                        "type": "temperature",
                        "unit": "°C",
                        "range": [50.0, 150.0],
                        "nominal_value": 100.0,
                        "unit_operation": "R-101",
                        "description": "Reactor temperature"
                    }
                ],
                "manipulated_variables": [
                    {
                        "name": "F_coolant",
                        "type": "flow",
                        "unit": "kg/h",
                        "range": [0.0, 5000.0],
                        "nominal_value": 2500.0,
                        "unit_operation": "R-101",
                        "description": "Coolant flow rate"
                    }
                ],
                "gain_matrix": [[0.9]]
            }
            st.download_button(
                "Download Template",
                json.dumps(template, indent=2),
                "pfd_template.json",
                "application/json",
                use_container_width=True
            )
    
    # Display process summary
    if st.session_state.pfd_data is not None:
        st.markdown("---")
        st.markdown("### 📊 Process Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Controlled Variables",
                len(st.session_state.pfd_data['controlled_variables'])
            )
        with col2:
            st.metric(
                "🎮 Manipulated Variables",
                len(st.session_state.pfd_data['manipulated_variables'])
            )
        with col3:
            st.metric(
                "⚙️ Unit Operations",
                len(st.session_state.pfd_data['unit_operations'])
            )
        with col4:
            dof = (len(st.session_state.pfd_data['manipulated_variables']) - 
                   len(st.session_state.pfd_data['controlled_variables']))
            st.metric("📐 Degrees of Freedom", dof)
        
        # Display variables in tables
        st.markdown("#### Controlled Variables")
        cv_data = []
        for cv in st.session_state.pfd_data['controlled_variables']:
            cv_data.append({
                'Name': cv['name'],
                'Type': cv['type'],
                'Unit': cv['unit'],
                'Range': f"[{cv['range'][0]}, {cv['range'][1]}]",
                'Nominal': cv['nominal_value'],
                'Unit Operation': cv['unit_operation']
            })
        st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)
        
        st.markdown("#### Manipulated Variables")
        mv_data = []
        for mv in st.session_state.pfd_data['manipulated_variables']:
            mv_data.append({
                'Name': mv['name'],
                'Type': mv['type'],
                'Unit': mv['unit'],
                'Range': f"[{mv['range'][0]}, {mv['range'][1]}]",
                'Nominal': mv['nominal_value'],
                'Unit Operation': mv['unit_operation']
            })
        st.dataframe(pd.DataFrame(mv_data), use_container_width=True, hide_index=True)
        
        # Display gain matrix
        if st.session_state.gain_matrix is not None and st.session_state.gain_matrix.size > 0:
            st.markdown("#### 📈 Steady-State Gain Matrix")
            
            cv_names = [cv['name'] for cv in st.session_state.pfd_data['controlled_variables']]
            mv_names = [mv['name'] for mv in st.session_state.pfd_data['manipulated_variables']]
            
            # Create DataFrame
            df_gain = pd.DataFrame(
                st.session_state.gain_matrix,
                columns=mv_names,
                index=cv_names
            )
            
            # Create heatmap
            fig = px.imshow(
                st.session_state.gain_matrix,
                labels=dict(x="Manipulated Variables", y="Controlled Variables", color="Gain"),
                x=mv_names,
                y=cv_names,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto",
                text_auto='.3f'
            )
            fig.update_layout(
                height=400,
                title="Gain Matrix Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(
                df_gain.style.background_gradient(cmap='RdBu_r', axis=None).format("{:.4f}"),
                use_container_width=True
            )

# ==================== TAB 2: RUN ANALYSIS ====================
with tab2:
    st.markdown('<div class="sub-header">🔄 Run Control Loop Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.pfd_data is None:
        st.markdown('<div class="warning-box">⚠️ Please load PFD data in the "Input Data" tab first.</div>', 
                   unsafe_allow_html=True)
    else:
        # Display process info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Controlled Variables",
                len(st.session_state.pfd_data['controlled_variables']),
                delta=None
            )
        with col2:
            st.metric(
                "🎮 Manipulated Variables",
                len(st.session_state.pfd_data['manipulated_variables']),
                delta=None
            )
        with col3:
            st.metric(
                "⚙️ Unit Operations",
                len(st.session_state.pfd_data['unit_operations']),
                delta=None
            )
        with col4:
            matrix_size = st.session_state.gain_matrix.shape
            st.metric(
                "📊 Gain Matrix",
                f"{matrix_size[0]}×{matrix_size[1]}",
                delta=None
            )
        
        st.markdown("---")
        
        # Analysis description
        st.markdown("### 🤖 Multi-Agent Analysis Pipeline")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            The analysis will proceed through the following agents:
            
            1. **🔍 PFD Analyzer Agent** - Analyzes process structure and identifies control requirements
            2. **📊 RGA Calculator Agent** - Computes Relative Gain Array for variable pairing
            3. **📈 Controllability Agent** - Performs SVD-based controllability assessment
            4. **🎯 Pairing Optimizer Agent** - Synthesizes optimal control loop pairings
            5. **✅ Validation Agent** - Validates control structure against engineering principles
            
            Each agent uses **Google Gemini AI** combined with classical control theory.
            """)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Analysis Methods:**")
            st.markdown("- Relative Gain Array (RGA)")
            st.markdown("- SVD Controllability")
            st.markdown("- Interaction Minimization")
            st.markdown("- Chemical Eng. Heuristics")
            st.markdown("- Bristol's Rules")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Run button
        if st.button("🚀 Run Control Loop Prediction", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.analysis_running):
            
            st.session_state.analysis_running = True
            st.session_state.agent_messages = []
            
            # Progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            agent_activity = st.empty()
            
            try:
                # Initialize workflow
                with progress_placeholder.container():
                    progress_bar = st.progress(0)
                    status_placeholder.info("🔧 Initializing workflow...")
                
                workflow = ControlLoopWorkflow(config)
                progress_bar.progress(10)
                
                # Run workflow with progress updates
                status_placeholder.info("🔍 Step 1/5: Analyzing PFD structure...")
                progress_bar.progress(20)
                
                # Create a callback to update progress (simulated)
                import time
                
                # Run the workflow
                result = workflow.run(
                    st.session_state.pfd_data,
                    st.session_state.gain_matrix
                )
                
                # Update progress incrementally
                for i, step in enumerate([
                    "📊 Step 2/5: Calculating RGA matrix...",
                    "📈 Step 3/5: Analyzing controllability...",
                    "🎯 Step 4/5: Optimizing pairings...",
                    "✅ Step 5/5: Validating control structure..."
                ], start=2):
                    progress_bar.progress(20 * i)
                    status_placeholder.info(step)
                    time.sleep(0.5)
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Analysis complete!")
                
                # Store results
                st.session_state.workflow_result = result
                st.session_state.analysis_running = False
                
                # Success message
                st.success("🎉 Control loop prediction completed successfully!")
                st.balloons()
                
                # Show quick summary
                if result.get('pairings'):
                    st.info(f"✨ Found {len(result['pairings'])} control loop pairings. Check the Results tab for details.")
                
            except Exception as e:
                st.session_state.analysis_running = False
                st.error(f"❌ Error during analysis: {str(e)}")
                logger.error(f"Workflow error: {e}", exc_info=True)
                
                with st.expander("🔍 Error Details"):
                    st.code(str(e))

# ==================== TAB 3: RESULTS ====================
with tab3:
    st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    if st.session_state.workflow_result is None:
        st.markdown('<div class="info-box">ℹ️ No results yet. Run the analysis in the "Run Analysis" tab.</div>', 
                   unsafe_allow_html=True)
    else:
        result = st.session_state.workflow_result
        
        # Check for errors
        if 'error' in result:
            st.error(f"❌ Analysis failed: {result['error']}")
        else:
            # Summary metrics
            st.markdown("### 📈 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔗 Control Loops",
                    len(result.get('pairings', [])),
                    help="Number of control loop pairings identified"
                )
            with col2:
                confidence = result.get('confidence_score', 0)
                st.metric(
                    "🎯 Confidence Score",
                    f"{confidence:.1%}",
                    delta=f"{(confidence - 0.7) * 100:.1f}%" if confidence > 0.7 else None,
                    help="Overall confidence in the control structure"
                )
            with col3:
                interaction_idx = result.get('interaction_index', 0)
                st.metric(
                    "🔄 Interaction Index",
                    f"{interaction_idx:.3f}",
                    delta="Good" if interaction_idx < 0.3 else ("Moderate" if interaction_idx < 0.5 else "High"),
                    delta_color="inverse",
                    help="Measure of loop interactions (lower is better)"
                )
            with col4:
                cond_num = result.get('condition_number', 0)
                st.metric(
                    "📊 Condition Number",
                    f"{cond_num:.2f}",
                    delta="Well-conditioned" if cond_num < 10 else ("Moderate" if cond_num < 100 else "Ill-conditioned"),
                    delta_color="inverse",
                    help="System conditioning (lower is better)"
                )
            
            st.markdown("---")
            
            # Control pairings
            st.markdown("### 🔗 Recommended Control Loop Pairings")
            
            pairings = result.get('pairings', [])
            if pairings:
                for i, pairing in enumerate(pairings, 1):
                    with st.expander(
                        f"**Loop {i}**: {pairing.get('controlled_variable', 'N/A')} ← {pairing.get('manipulated_variable', 'N/A')} "
                        f"[{pairing.get('controller_type', 'PID')}] (Confidence: {pairing.get('overall_confidence', 0):.1%})",
                        expanded=True
                    ):
                        col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"σ{i+1}" for i in range(len(singular_values))],
                        y=singular_values,
                        marker_color='lightblue',
                        text=[f"{sv:.4f}" for sv in singular_values],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        xaxis_title="Singular Value",
                        yaxis_title="Magnitude",
                        yaxis_type="log",
                        height=350,
                        title="Singular Value Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Controllability Assessment:**")
                    
                    cond_num = result.get('condition_number', 0)
                    if cond_num < 10:
                        st.success("✅ Well-conditioned system")
                    elif cond_num < 100:
                        st.warning("⚠️ Moderately conditioned")
                    else:
                        st.error("❌ Ill-conditioned system")
                    
                    st.metric("Condition Number", f"{cond_num:.2f}")
                    st.metric("Smallest σ", f"{singular_values[-1]:.4f}")
                    st.metric("Largest σ", f"{singular_values[0]:.4f}")
                    
                    # Controllability interpretation
                    st.markdown("---")
                    st.markdown("**Interpretation:**")
                    st.caption("Large σ: Strong control direction")
                    st.caption("Small σ: Weak control direction")
            
            st.markdown("---")
            
            # Validation Results
            validation = result.get('validation_results', {})
            if validation:
                st.markdown("### ✅ Validation Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    status = validation.get('overall_status', 'UNKNOWN')
                    if status == 'APPROVED':
                        st.success(f"**Overall Status:** {status} ✅")
                    elif status == 'CONDITIONAL':
                        st.warning(f"**Overall Status:** {status} ⚠️")
                    else:
                        st.error(f"**Overall Status:** {status} ❌")
                    
                    st.markdown("**Validation Checks:**")
                    checks = {
                        'Safety': validation.get('safety_check', 'N/A'),
                        'Engineering': validation.get('engineering_check', 'N/A'),
                        'Performance': validation.get('performance_check', 'N/A'),
                        'Operational': validation.get('operational_check', 'N/A')
                    }
                    
                    for check_name, check_status in checks.items():
                        if check_status == 'PASS':
                            st.markdown(f"- **{check_name}:** ✅ {check_status}")
                        else:
                            st.markdown(f"- **{check_name}:** ⚠️ {check_status}")
                
                with col2:
                    st.markdown("**Summary:**")
                    st.info(validation.get('summary', 'No summary available'))
            
            # Warnings
            warnings = result.get('warnings', [])
            if warnings:
                st.markdown("### ⚠️ Warnings")
                for warning in warnings:
                    st.warning(warning)
            
            # Recommendations
            recommendations = result.get('recommendations', [])
            if recommendations:
                st.markdown("### 💡 Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"**{i}.** {rec}")
            
            st.markdown("---")
            
            # Download section
            st.markdown("### 📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON download
                st.download_button(
                    "📄 Download JSON",
                    json.dumps(result, indent=2, default=str),
                    "control_structure_results.json",
                    "application/json",
                    use_container_width=True,
                    help="Download complete results as JSON"
                )
            
            with col2:
                # Create CSV of pairings
                if pairings:
                    pairing_df = pd.DataFrame([
                        {
                            'CV': p.get('controlled_variable'),
                            'MV': p.get('manipulated_variable'),
                            'Controller': p.get('controller_type'),
                            'RGA': p.get('rga_value'),
                            'Confidence': p.get('overall_confidence')
                        }
                        for p in pairings
                    ])
                    csv = pairing_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        "control_pairings.csv",
                        "text/csv",
                        use_container_width=True,
                        help="Download pairings as CSV"
                    )
            
            with col3:
                # Create markdown report
                report = f"""# Control Structure Analysis Report

## Process: {st.session_state.pfd_data['name']}

{st.session_state.pfd_data['description']}

## Summary Metrics
- **Control Loops:** {len(pairings)}
- **Confidence Score:** {result.get('confidence_score', 0):.1%}
- **Interaction Index:** {result.get('interaction_index', 0):.3f}
- **Condition Number:** {result.get('condition_number', 0):.2f}

## Control Loop Pairings

"""
                for i, pairing in enumerate(pairings, 1):
                    report += f"""
### Loop {i}: {pairing.get('controlled_variable')} ← {pairing.get('manipulated_variable')}

- **Controller Type:** {pairing.get('controller_type')}
- **RGA Value:** {pairing.get('rga_value', 0):.3f}
- **Confidence:** {pairing.get('overall_confidence', 0):.1%}

**Reasoning:** {pairing.get('reasoning', 'N/A')}

**Chemical Engineering Rationale:** {pairing.get('chemical_eng_rationale', 'N/A')}

**Tuning Guidance:** {pairing.get('tuning_guidance', 'N/A')}

---
"""
                
                report += "\n## Recommendations\n\n"
                for i, rec in enumerate(recommendations, 1):
                    report += f"{i}. {rec}\n"
                
                if warnings:
                    report += "\n## Warnings\n\n"
                    for warning in warnings:
                        report += f"- {warning}\n"
                
                st.download_button(
                    "📝 Download Report",
                    report,
                    "control_structure_report.md",
                    "text/markdown",
                    use_container_width=True,
                    help="Download detailed report as Markdown"
                )

# ==================== TAB 4: AGENT ACTIVITY ====================
with tab4:
    st.markdown('<div class="sub-header">🤖 Agent Activity Log</div>', unsafe_allow_html=True)
    
    if st.session_state.workflow_result is None:
        st.markdown('<div class="info-box">ℹ️ No agent activity yet. Run the analysis first.</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown("""
        This section shows the activity and outputs from each agent in the multi-agent pipeline.
        Each agent contributes specialized analysis to the final control structure recommendation.
        """)
        
        # Get messages from result if available
        messages = st.session_state.workflow_result.get('messages', [])
        
        if messages:
            st.markdown("### 📨 Agent Messages")
            
            for msg in messages:
                agent_name = msg.get('agent', 'Unknown')
                content = msg.get('content', 'No content')
                
                with st.expander(f"🤖 {agent_name}", expanded=False):
                    st.markdown(f'<div class="agent-message">{content}</div>', 
                               unsafe_allow_html=True)
        
        # Display detailed analysis if available
        result = st.session_state.workflow_result
        
        if result.get('pfd_analysis'):
            with st.expander("🔍 PFD Analysis (Detailed)", expanded=False):
                st.markdown(result['pfd_analysis'])
        
        if result.get('rga_analysis'):
            with st.expander("📊 RGA Analysis (Detailed)", expanded=False):
                st.markdown(result['rga_analysis'])
        
        if result.get('controllability_analysis'):
            with st.expander("📈 Controllability Analysis (Detailed)", expanded=False):
                st.markdown(result['controllability_analysis'])
        
        if result.get('pairing_reasoning'):
            with st.expander("🎯 Pairing Optimization (Detailed)", expanded=False):
                st.markdown(result['pairing_reasoning'])
        
        # Show errors if any
        errors = result.get('errors', [])
        if errors:
            st.markdown("### ❌ Errors")
            for error in errors:
                st.error(error)

# ==================== TAB 5: HELP ====================
with tab5:
    st.markdown('<div class="sub-header">📖 Help & Documentation</div>', unsafe_allow_html=True)
    
    with st.expander("🎯 About This Tool", expanded=True):
        st.markdown("""
        ## PFD Control Loop Prediction System
        
        This tool uses **AI-powered multi-agent system** to predict optimal control structures 
        for Process Flow Diagrams (PFDs). It combines:
        
        - **Classical Control Theory** (RGA, SVD, Interaction Analysis)
        - **Chemical Engineering Principles** (Unit operation heuristics, process knowledge)
        - **AI Reasoning** (Google Gemini for intelligent decision-making)
        - **LangGraph Workflow** (Multi-agent orchestration)
        
        ### Key Features
        
        ✅ **RGA Analysis** - Relative Gain Array for variable pairing recommendations  
        ✅ **SVD Controllability** - Singular Value Decomposition for system assessment  
        ✅ **Interaction Minimization** - Identifies and reduces loop coupling  
        ✅ **Chemical Engineering Heuristics** - Domain-specific control strategies  
        ✅ **Multi-Agent Architecture** - Specialized agents for comprehensive analysis  
        ✅ **Validation Engine** - Safety and performance validation  
        """)
    
    with st.expander("🔧 How It Works"):
        st.markdown("""
        ## Multi-Agent Workflow
        
        ### 1. PFD Analyzer Agent 🔍
        - Analyzes process structure and topology
        - Identifies unit operations and their characteristics
        - Determines control objectives and priorities
        - Applies chemical engineering fundamentals
        
        ### 2. RGA Calculator Agent 📊
        - Computes Relative Gain Array: `RGA = G ⊙ (G⁻¹)ᵀ`
        - Identifies potential CV-MV pairings
        - Applies Bristol's rules for pairing
        - Detects problematic interactions
        
        ### 3. Controllability Analyzer Agent 📈
        - Performs SVD: `G = U Σ Vᵀ`
        - Calculates condition number and singular values
        - Assesses system controllability
        - Validates RGA pairings against dominant directions
        
        ### 4. Pairing Optimizer Agent 🎯
        - Integrates RGA, SVD, and interaction metrics
        - Applies chemical engineering heuristics
        - Recommends controller types (PI, PID, Cascade, etc.)
        - Optimizes using multiple criteria
        
        ### 5. Validation Agent ✅
        - Performs safety validation
        - Checks engineering feasibility
        - Assesses expected performance
        - Provides final recommendations
        """)
    
    with st.expander("📊 Understanding the Metrics"):
        st.markdown("""
        ## Key Metrics Explained
        
        ### Relative Gain Array (RGA)
        
        The RGA element λᵢⱼ indicates how CV_i responds to MV_j:
        
        - **λᵢⱼ ≈ 1.0**: ✅ Excellent pairing (ideal)
        - **0.5 < λᵢⱼ < 1.5**: 🟢 Good pairing
        - **0 < λᵢⱼ < 0.5**: 🟡 Poor pairing (weak effect)
        - **λᵢⱼ < 0**: 🔴 Bad pairing (avoid! can cause instability)
        
        ### Condition Number (κ)
        
        Measures system sensitivity: `κ = σₘₐₓ / σₘᵢₙ`
        
        - **κ < 10**: ✅ Well-conditioned (easy to control)
        - **10 < κ < 100**: 🟡 Moderately conditioned
        - **κ > 100**: 🔴 Ill-conditioned (difficult to control)
        
        ### Interaction Index (I)
        
        Measures loop coupling: `I = ||G - diag(G)|| / ||G||`
        
        - **I < 0.3**: ✅ Low interaction (decentralized control OK)
        - **0.3 < I < 0.5**: 🟡 Moderate interaction (careful tuning)
        - **I > 0.5**: 🔴 High interaction (consider MPC)
        
        ### Confidence Score
        
        Overall confidence in the control structure (0-100%):
        
        - Combines RGA quality, controllability, and validation results
        - Higher is better
        - > 80% indicates high confidence
        """)
    
    with st.expander("📋 Input Data Format"):
        st.markdown("""
        ## JSON Data Structure
        
        Your PFD data should follow this structure:
        
        ```json
        {
          "name": "Process Name",
          "description": "Detailed process description",
          "unit_operations": [
            {
              "name": "R-101",
              "type": "reactor",
              "description": "Main reactor"
            }
          ],
          "controlled_variables": [
            {
              "name": "T_reactor",
              "type": "temperature",
              "unit": "°C",
              "range": [50.0, 150.0],
              "nominal_value": 100.0,
              "unit_operation": "R-101",
              "description": "Reactor temperature"
            }
          ],
          "manipulated_variables": [
            {
              "name": "F_coolant",
              "type": "flow",
              "unit": "kg/h",
              "range": [0.0, 5000.0],
              "nominal_value": 2500.0,
              "unit_operation": "R-101",
              "description": "Coolant flow rate"
            }
          ],
          "gain_matrix": [
            [0.9]
          ]
        }
        ```
        
        ### Required Fields
        - `name`: Process name (string)
        - `description`: Process description (string)
        - `unit_operations`: Array of unit operations
        - `controlled_variables`: Array of CVs with properties
        - `manipulated_variables`: Array of MVs with properties
        - `gain_matrix`: 2D array (n_CVs × n_MVs)
        
        ### Optional Fields
        - `disturbance_variables`: Array of disturbances
        - `time_constants`: 2D array of time constants
        """)
    
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ## FAQ
        
        **Q: What is the Relative Gain Array (RGA)?**  
        A: RGA is a matrix that shows how controlled and manipulated variables interact. 
        It helps identify good pairings for decentralized control.
        
        **Q: What's a good condition number?**  
        A: < 10 is excellent, 10-100 is acceptable, > 100 indicates the system is 
        ill-conditioned and may be difficult to control.
        
        **Q: How do I interpret the interaction index?**  
        A: < 0.3 means low interaction (good for decentralized control), 
        0.3-0.5 is moderate (careful tuning needed), > 0.5 is high 
        (consider advanced control like MPC).
        
        **Q: What if I get negative RGA values?**  
        A: Negative RGA values indicate that pairing should be avoided as it can 
        lead to instability in decentralized control.
        
        **Q: Can I use this for non-square systems?**  
        A: Yes, but the system will use pseudo-inverse for RGA calculation. 
        You'll have more MVs than CVs (degrees of freedom for optimization).
        
        **Q: How accurate is the AI analysis?**  
        A: The system combines proven control theory with AI reasoning. 
        Always validate recommendations with process knowledge and dynamic simulation.
        
        **Q: What should I do with the recommendations?**  
        A: Use them as a starting point for control system design. 
        Perform dynamic simulation, tune controllers, and validate against 
        process requirements before implementation.
        """)
    
    with st.expander("🔗 Additional Resources"):
        st.markdown("""
        ## Learning Resources
        
        ### Control Theory
        - [Introduction to Process Control](https://en.wikipedia.org/wiki/Process_control)
        - [Relative Gain Array](https://en.wikipedia.org/wiki/Relative_gain_array)
        - [SVD and Controllability](https://en.wikipedia.org/wiki/Controllability)
        
        ### Chemical Engineering
        - Luyben, W.L., et al. (1997). "Plantwide Process Control"
        - Skogestad, S. (2004). "Control Structure Design"
        - Stephanopoulos, G. (1984). "Chemical Process Control"
        
        ### AI and LangGraph
        - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
        - [LangChain Documentation](https://python.langchain.com/)
        - [Google Gemini API](https://ai.google.dev/)
        
        ### Tools
        - [MATLAB Control Toolbox](https://www.mathworks.com/products/control.html)
        - [Python Control Systems Library](https://python-control.readthedocs.io/)
        """)
    
    with st.expander("🐛 Troubleshooting"):
        st.markdown("""
        ## Common Issues
        
        **API Key Error**
        - Ensure GOOGLE_API_KEY is set in your .env file
        - Check that the API key is valid and has sufficient quota
        
        **Data Validation Errors**
        - Check that gain matrix dimensions match (n_CVs × n_MVs)
        - Ensure all required fields are present
        - Verify nominal values are within specified ranges
        
        **Analysis Failures**
        - Check system logs for detailed error messages
        - Verify gain matrix is not singular
        - Ensure reasonable values in gain matrix (not too large/small)
        
        **Poor Results**
        - Verify gain matrix accuracy
        - Check for proper scaling of variables
        - Review process description for accuracy
        - Consider providing time constants for better analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>PFD Control Loop Prediction System</strong></p>
    <p>Powered by LangGraph, Google Gemini, and Chemical Engineering Principles</p>
    <p>Version 1.0.0 | © 2024</p>
</div>
""", unsafe_allow_html=True)