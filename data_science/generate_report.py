"""
Generate comprehensive PDF report for the Jadugora Uranium Mining Impact Assessment project.
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from weasyprint import HTML, CSS
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/report_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("reports/final", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def create_html_report():
    """
    Create HTML report using Jinja2 templates.
    
    Returns:
        str: Path to the generated HTML file
    """
    logger.info("Creating HTML report...")
    
    # Create templates directory and template file
    os.makedirs("reports/templates", exist_ok=True)
    
    # Create base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            text-align: center;
            font-size: 24pt;
            margin-bottom: 1em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            font-size: 20pt;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 {
            font-size: 16pt;
        }
        h4 {
            font-size: 14pt;
        }
        p, ul, ol {
            font-size: 12pt;
            text-align: justify;
        }
        .figure {
            margin: 20px 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            height: auto;
        }
        .figure-caption {
            font-style: italic;
            font-size: 10pt;
            color: #7f8c8d;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header img {
            max-width: 150px;
            height: auto;
        }
        .date {
            text-align: right;
            font-style: italic;
            margin-bottom: 30px;
        }
        .toc {
            background-color: #f8f9fa;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .toc h2 {
            margin-top: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc ul ul {
            padding-left: 20px;
        }
        .toc a {
            text-decoration: none;
            color: #3498db;
        }
        .toc a:hover {
            text-decoration: underline;
        }
        .page-break {
            page-break-after: always;
        }
        .reference {
            font-size: 10pt;
            margin-left: 20px;
            text-indent: -20px;
        }
        .appendix {
            margin-top: 40px;
        }
        .code {
            font-family: monospace;
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 10pt;
            white-space: pre-wrap;
        }
        .highlight {
            background-color: #ffffcc;
            padding: 2px 5px;
            border-radius: 3px;
        }
        .footer {
            text-align: center;
            font-size: 10pt;
            color: #7f8c8d;
            margin-top: 40px;
            border-top: 1px solid #bdc3c7;
            padding-top: 10px;
        }
        @page {
            size: A4;
            margin: 2cm;
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 10pt;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p><strong>{{ subtitle }}</strong></p>
    </div>
    
    <div class="date">
        <p>{{ date }}</p>
    </div>
    
    {% block content %}{% endblock %}
    
    <div class="footer">
        <p>Jadugora Uranium Mining Impact Assessment Project</p>
    </div>
</body>
</html>
"""
    
    # Create report template
    report_template = """{% extends "base.html" %}

{% block content %}
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#introduction">1. Introduction</a>
                <ul>
                    <li><a href="#background">1.1 Background</a></li>
                    <li><a href="#objectives">1.2 Project Objectives</a></li>
                    <li><a href="#scope">1.3 Scope and Limitations</a></li>
                </ul>
            </li>
            <li><a href="#methodology">2. Methodology</a>
                <ul>
                    <li><a href="#data-collection">2.1 Data Collection</a></li>
                    <li><a href="#data-processing">2.2 Data Processing</a></li>
                    <li><a href="#analysis-approach">2.3 Analysis Approach</a></li>
                    <li><a href="#modeling-approach">2.4 Modeling Approach</a></li>
                </ul>
            </li>
            <li><a href="#exploratory-analysis">3. Exploratory Data Analysis</a>
                <ul>
                    <li><a href="#environmental-data">3.1 Environmental Data</a></li>
                    <li><a href="#health-data">3.2 Health Data</a></li>
                    <li><a href="#socioeconomic-data">3.3 Socioeconomic Data</a></li>
                    <li><a href="#mining-data">3.4 Mining Production Data</a></li>
                </ul>
            </li>
            <li><a href="#modeling-results">4. Modeling Results</a>
                <ul>
                    <li><a href="#environmental-impact">4.1 Environmental Impact Models</a></li>
                    <li><a href="#health-impact">4.2 Health Impact Models</a></li>
                    <li><a href="#future-projections">4.3 Future Projections</a></li>
                </ul>
            </li>
            <li><a href="#mlops-implementation">5. MLOps Implementation</a>
                <ul>
                    <li><a href="#pipeline-architecture">5.1 Pipeline Architecture</a></li>
                    <li><a href="#model-registry">5.2 Model Registry and Versioning</a></li>
                    <li><a href="#deployment">5.3 Deployment Strategy</a></li>
                    <li><a href="#monitoring">5.4 Monitoring and Maintenance</a></li>
                </ul>
            </li>
            <li><a href="#findings">6. Key Findings and Insights</a>
                <ul>
                    <li><a href="#environmental-findings">6.1 Environmental Impacts</a></li>
                    <li><a href="#health-findings">6.2 Health Impacts</a></li>
                    <li><a href="#socioeconomic-findings">6.3 Socioeconomic Impacts</a></li>
                </ul>
            </li>
            <li><a href="#recommendations">7. Recommendations</a>
                <ul>
                    <li><a href="#policy-recommendations">7.1 Policy Recommendations</a></li>
                    <li><a href="#mitigation-strategies">7.2 Mitigation Strategies</a></li>
                    <li><a href="#future-research">7.3 Future Research Directions</a></li>
                </ul>
            </li>
            <li><a href="#conclusion">8. Conclusion</a></li>
            <li><a href="#references">References</a></li>
            <li><a href="#appendix">Appendix</a>
                <ul>
                    <li><a href="#appendix-a">A. Data Dictionary</a></li>
                    <li><a href="#appendix-b">B. Model Performance Metrics</a></li>
                    <li><a href="#appendix-c">C. Code Repository Structure</a></li>
                </ul>
            </li>
        </ul>
    </div>
    
    <div class="page-break"></div>
    
    <section id="executive-summary">
        <h2>Executive Summary</h2>
        <p>This report presents a comprehensive data science project focused on analyzing the environmental, local, and societal consequences of uranium mining in Jadugora, Jharkhand, India. Using advanced data analysis and machine learning techniques, we have developed models to assess the relationships between mining activities and various impact indicators, including environmental contamination, health outcomes, and socioeconomic factors.</p>
        
        <p>Key findings from our analysis include:</p>
        <ul>
            <li>Strong correlations between mining production metrics and environmental contamination levels, particularly for radiation and heavy metal concentrations in soil and water</li>
            <li>Significant associations between environmental contamination indicators and health outcomes in nearby communities, with elevated rates of certain diseases in proximity to mining operations</li>
            <li>Complex socioeconomic impacts, including both employment opportunities and community displacement</li>
            <li>Projections suggesting continued environmental and health impacts if current mining practices continue without additional mitigation measures</li>
        </ul>
        
        <p>The project implements a complete MLOps pipeline for model training, evaluation, and deployment, enabling ongoing monitoring and updates as new data becomes available. This infrastructure supports evidence-based decision-making for policymakers, environmental agencies, and community organizations working to address the challenges faced by affected communities.</p>
        
        <p>Recommendations include enhanced environmental monitoring, implementation of additional safety measures, community health programs, and further research into remediation technologies. These measures could significantly reduce the negative impacts while maintaining the economic benefits of mining operations.</p>
    </section>
    
    <div class="page-break"></div>
    
    <section id="introduction">
        <h2>1. Introduction</h2>
        
        <section id="background">
            <h3>1.1 Background</h3>
            <p>Uranium mining has been conducted in Jadugora, Jharkhand, India since the 1960s, providing essential fuel for India's nuclear power program. The Uranium Corporation of India Limited (UCIL) operates these mines, which are among the oldest uranium mining operations in the country. While these operations contribute significantly to India's energy security and economic development, concerns have been raised about their environmental and health impacts on local communities.</p>
            
            <p>The documentary "Buddha Weeps in Jadugora" (1999) by filmmaker Shriprakash brought international attention to these issues, highlighting reports of elevated radiation levels, contaminated water sources, and increased incidence of health problems in communities surrounding the mines. Despite the controversy, comprehensive scientific studies examining the relationships between mining activities and various impact indicators have been limited.</p>
            
            <p>This project aims to address this gap by applying data science and machine learning techniques to analyze available data and develop predictive models that can help understand and quantify these impacts. By creating a robust analytical framework, we seek to provide evidence-based insights that can inform policy decisions and mitigation strategies.</p>
        </section>
        
        <section id="objectives">
            <h3>1.2 Project Objectives</h3>
            <p>The primary objectives of this data science project are:</p>
            <ul>
                <li>To collect, process, and analyze data related to uranium mining operations in Jadugora and their potential environmental, health, and socioeconomic impacts</li>
                <li>To develop machine learning models that can quantify relationships between mining activities and various impact indicators</li>
                <li>To create predictive models for assessing future environmental and health impacts under different scenarios</li>
                <li>To implement a cloud-based MLOps pipeline for model deployment, monitoring, and maintenance</li>
                <li>To provide data-driven recommendations for policy interventions and mitigation strategies</li>
            </ul>
        </section>
        
        <section id="scope">
            <h3>1.3 Scope and Limitations</h3>
            <p>This project encompasses the following scope:</p>
            <ul>
                <li>Analysis of environmental data including radiation levels, water quality, and soil contamination in the Jadugora region</li>
                <li>Assessment of health data including disease prevalence, birth defects, and mortality rates in communities near mining operations</li>
                <li>Examination of socioeconomic factors including employment, education, and displacement</li>
                <li>Development of predictive models for environmental contamination and health outcomes</li>
                <li>Implementation of a complete MLOps pipeline for model deployment and monitoring</li>
            </ul>
            
            <p>Key limitations of this study include:</p>
            <ul>
                <li>Reliance on synthetic data that simulates real-world patterns due to limited availability of comprehensive field measurements</li>
                <li>Challenges in establishing causality versus correlation in observed relationships</li>
                <li>Limited temporal and spatial resolution in available datasets</li>
                <li>Potential confounding factors not captured in the available data</li>
                <li>Technical limitations in the model registration process for environmental impact models</li>
            </ul>
        </section>
    </section>
    
    <div class="page-break"></div>
    
    <section id="methodology">
        <h2>2. Methodology</h2>
        
        <section id="data-collection">
            <h3>2.1 Data Collection</h3>
            <p>The data collection process involved gathering information from multiple sources to create a comprehensive dataset covering environmental, health, socioeconomic, and mining production aspects. For this project, we generated synthetic datasets that simulate real-world patterns based on available literature and reports about uranium mining impacts.</p>
            
            <p>The following datasets were collected:</p>
            <ul>
                <li><strong>Environmental Data</strong>: Radiation levels, water quality parameters (pH, heavy metals, uranium concentration), and soil contamination measurements (uranium, radium, lead, arsenic)</li>
                <li><strong>Health Data</strong>: Disease prevalence (cancer, respiratory diseases, skin disorders, kidney diseases), birth defects, and mortality rates</li>
                <li><strong>Socioeconomic Data</strong>: Employment statistics, 
(Content truncated due to size limit. Use line ranges to read in chunks)