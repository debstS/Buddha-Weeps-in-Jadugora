import React from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

const Methodology: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Methodology</h1>
      
      <Tabs defaultValue="data-collection" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="data-collection">Data Collection</TabsTrigger>
          <TabsTrigger value="data-processing">Data Processing</TabsTrigger>
          <TabsTrigger value="analysis">Analysis Approach</TabsTrigger>
          <TabsTrigger value="modeling">Modeling Approach</TabsTrigger>
        </TabsList>
        
        <TabsContent value="data-collection" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Data Collection</h2>
          <p className="mb-4">
            The data collection process involved gathering information from multiple sources to create a comprehensive dataset 
            covering environmental, health, socioeconomic, and mining production aspects. For this project, we generated synthetic 
            datasets that simulate real-world patterns based on available literature and reports about uranium mining impacts.
          </p>
          
          <p className="mb-4">The following datasets were collected:</p>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li><strong>Environmental Data</strong>: Radiation levels, water quality parameters (pH, heavy metals, uranium concentration), and soil contamination measurements (uranium, radium, lead, arsenic)</li>
            <li><strong>Health Data</strong>: Disease prevalence (cancer, respiratory diseases, skin disorders, kidney diseases), birth defects, and mortality rates</li>
            <li><strong>Socioeconomic Data</strong>: Employment statistics, education levels, and displacement information</li>
            <li><strong>Mining Production Data</strong>: Ore extraction volumes, uranium production, waste generation, tailings volume, and water usage</li>
            <li><strong>Spatial Data</strong>: Mining sites, villages, and environmental sampling points</li>
          </ul>
          
          <p>
            Data collection was implemented using a modular Python script that generates consistent, interrelated datasets with 
            realistic temporal and spatial patterns. This approach allowed us to create a controlled environment for analysis 
            while maintaining realistic relationships between variables.
          </p>
        </TabsContent>
        
        <TabsContent value="data-processing" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Data Processing</h2>
          <p className="mb-4">
            Raw data underwent several processing steps to prepare it for analysis and modeling:
          </p>
          
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li><strong>Data Cleaning</strong>: Handling missing values, removing outliers, and correcting inconsistencies</li>
            <li><strong>Feature Engineering</strong>: Creating derived features such as distance from mining sites, years of operation, and cumulative exposure metrics</li>
            <li><strong>Temporal Aggregation</strong>: Converting time-series data to annual averages for alignment with health and socioeconomic indicators</li>
            <li><strong>Spatial Integration</strong>: Linking environmental measurements with nearby communities for impact assessment</li>
            <li><strong>Data Merging</strong>: Combining datasets from different domains to create integrated analytical datasets</li>
            <li><strong>Column Standardization</strong>: Ensuring consistent naming conventions across merged datasets</li>
          </ul>
          
          <p>
            During the data processing phase, we encountered and resolved several challenges, including duplicate column names 
            in merged datasets and inconsistent temporal resolutions. These issues were addressed through custom data cleaning 
            scripts that standardized column names and temporal aggregation methods.
          </p>
        </TabsContent>
        
        <TabsContent value="analysis" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Analysis Approach</h2>
          <p className="mb-4">
            Our exploratory data analysis followed a systematic approach:
          </p>
          
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li><strong>Univariate Analysis</strong>: Examining the distribution and summary statistics of individual variables</li>
            <li><strong>Bivariate Analysis</strong>: Investigating relationships between pairs of variables, particularly between mining activities and impact indicators</li>
            <li><strong>Temporal Analysis</strong>: Tracking changes in key indicators over time and identifying trends</li>
            <li><strong>Spatial Analysis</strong>: Analyzing how impacts vary with distance from mining operations</li>
            <li><strong>Correlation Analysis</strong>: Quantifying the strength and direction of relationships between variables</li>
            <li><strong>Visualization</strong>: Creating informative plots and charts to communicate patterns and relationships</li>
          </ul>
          
          <p>
            We used Python's data science ecosystem, including pandas for data manipulation, matplotlib and seaborn for visualization, 
            and scipy for statistical analysis. All analyses were documented in structured reports with reproducible code.
          </p>
        </TabsContent>
        
        <TabsContent value="modeling" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Modeling Approach</h2>
          <p className="mb-4">
            We developed several types of machine learning models to address different aspects of the impact assessment:
          </p>
          
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li><strong>Environmental Impact Models</strong>: Predicting environmental contamination levels based on mining activities</li>
            <li><strong>Health Impact Models</strong>: Predicting health outcomes based on environmental factors and mining activities</li>
            <li><strong>Future Projection Models</strong>: Forecasting future impacts based on historical trends and potential scenarios</li>
          </ul>
          
          <p className="mb-4">For each modeling task, we evaluated multiple algorithms including:</p>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>Linear models (Linear Regression, Ridge, Lasso)</li>
            <li>Tree-based models (Random Forest, Gradient Boosting)</li>
            <li>Polynomial models for non-linear relationships</li>
          </ul>
          
          <p>
            Models were evaluated using cross-validation and metrics including RMSE, MAE, and R² score. Feature importance analysis 
            was conducted to identify the most influential factors for each outcome. The best-performing models were selected for 
            deployment in the MLOps pipeline.
          </p>
        </TabsContent>
      </Tabs>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Technical Implementation</h2>
          <p className="mb-4">
            The project was implemented using a modular, reproducible approach with the following technical components:
          </p>
          
          <ul className="list-disc pl-6 space-y-2">
            <li><strong>Programming Languages</strong>: Python for data processing, analysis, and modeling</li>
            <li><strong>Data Processing</strong>: Pandas, NumPy for data manipulation and numerical operations</li>
            <li><strong>Visualization</strong>: Matplotlib, Seaborn for static visualizations, Plotly for interactive charts</li>
            <li><strong>Machine Learning</strong>: Scikit-learn for model development and evaluation</li>
            <li><strong>Model Management</strong>: MLflow for experiment tracking, model registry, and versioning</li>
            <li><strong>API Development</strong>: Flask for creating model serving APIs</li>
            <li><strong>Monitoring</strong>: Dash for creating interactive monitoring dashboards</li>
            <li><strong>Cloud Deployment</strong>: AWS services for scalable, reliable deployment</li>
          </ul>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Ethical Considerations</h2>
          <p className="mb-4">
            Throughout the project, we maintained a strong focus on ethical considerations, including:
          </p>
          
          <ul className="list-disc pl-6 space-y-2">
            <li><strong>Data Privacy</strong>: Ensuring that all data used in the analysis was anonymized and aggregated to protect individual privacy</li>
            <li><strong>Transparency</strong>: Clearly documenting all methods, assumptions, and limitations to enable critical evaluation</li>
            <li><strong>Balanced Reporting</strong>: Presenting findings in a balanced manner that acknowledges both positive and negative impacts</li>
            <li><strong>Community Perspective</strong>: Incorporating community perspectives and concerns in the analysis and recommendations</li>
            <li><strong>Actionable Insights</strong>: Focusing on generating insights that can lead to practical improvements in environmental and health outcomes</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
};

export default Methodology;
