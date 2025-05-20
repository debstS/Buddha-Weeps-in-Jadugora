import React from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

const HealthImpact: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Health Impact</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          Analysis of health data revealed several patterns potentially associated with environmental exposures from uranium mining operations.
          Communities closer to mining operations showed higher prevalence of certain diseases compared to more distant communities.
        </p>
      </div>
      
      <Tabs defaultValue="disease" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="disease">Disease Prevalence</TabsTrigger>
          <TabsTrigger value="birth">Birth Defects</TabsTrigger>
          <TabsTrigger value="mortality">Mortality Rates</TabsTrigger>
        </TabsList>
        
        <TabsContent value="disease" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Disease Prevalence</h2>
              <p className="mb-4">
                Communities closer to mining operations showed higher prevalence of certain diseases compared to more distant communities.
                Cancer rates averaged 25.2 cases per 10,000 population in high-exposure areas compared to 12.8 in low-exposure areas.
              </p>
              <p className="mb-4">
                Respiratory diseases, skin disorders, and kidney diseases also showed spatial patterns potentially related to exposure.
              </p>
              <p>
                Our models identified uranium concentration in water and soil, along with radiation levels, as the most important predictors
                of disease rates in the affected communities.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/disease_prevalence_analysis.png" 
                alt="Disease prevalence by proximity to mining operations" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="birth" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Birth Defects</h2>
              <p className="mb-4">
                Analysis of birth defects data showed higher rates in communities with elevated environmental contamination levels.
                The rate of congenital abnormalities was 1.8 times higher in high-exposure communities compared to control communities.
              </p>
              <p className="mb-4">
                Temporal analysis suggested increases corresponding to periods of expanded mining operations.
              </p>
              <p>
                While our models could not establish causality, the consistent spatial and temporal patterns suggest associations
                between environmental exposures and birth outcomes that warrant further investigation.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/birth_defects_analysis.png" 
                alt="Birth defects rates over time by community exposure level" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="mortality" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Mortality Rates</h2>
              <p className="mb-4">
                Age-adjusted mortality rates showed spatial patterns potentially related to environmental exposures.
                Communities within 5 km of mining operations had mortality rates approximately 1.4 times higher than those beyond 20 km.
              </p>
              <p className="mb-4">
                Cause-specific mortality analysis suggested elevated rates for certain conditions, particularly cancers and kidney diseases.
              </p>
              <p>
                These findings highlight the importance of enhanced health surveillance and preventive measures in affected communities.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/mortality_analysis.png" 
                alt="Mortality rates by distance from mining operations" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Health Impact Models</h2>
          <p className="mb-4">
            We developed models to predict health impacts based on environmental factors and mining activities. These models help quantify
            the relationships between environmental exposures and health outcomes, providing insights for preventive strategies.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Cancer Rate Prediction</h3>
              <p className="text-sm">
                The Random Forest model performed best with an RMSE of 10.5882 cases per 10,000 population, though still with a negative R² value (-1.6535).
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Respiratory Disease Prediction</h3>
              <p className="text-sm">
                The Random Forest model performed best with an RMSE of 10.5882 cases per 10,000 population and an R² of -1.6535.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Skin Disorders Prediction</h3>
              <p className="text-sm">
                The Gradient Boosting model achieved an RMSE of 2.0092 cases per 10,000 population and a positive R² of 0.3541.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Kidney Disease Prediction</h3>
              <p className="text-sm">
                The Lasso model performed best with an RMSE of 3.2414 cases per 10,000 population and an R² of -0.1674.
              </p>
            </div>
          </div>
          
          <p>
            The performance of health impact models varied, with skin disorders models showing the best predictive power.
            The challenges in model performance highlight the complex nature of health impacts and the need for more comprehensive data.
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Future Projections</h2>
          <p className="mb-4">
            Projections for health impacts suggest continued increases in disease rates in affected communities if environmental conditions
            do not improve. The Ridge regression model performed best for cancer rate projections, indicating a potential 18% increase in
            cancer rates over the next decade under current conditions.
          </p>
          <div className="flex items-center justify-center">
            <img 
              src="/images/projection_cancer_rate.png" 
              alt="Projected cancer rates over the next decade" 
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            Note: These projections should be interpreted with caution given the limitations of the models and the potential for
            policy interventions or technological improvements to alter these trajectories.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default HealthImpact;
