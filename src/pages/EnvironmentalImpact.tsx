import React from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

const EnvironmentalImpact: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Environmental Impact</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          Our analysis of environmental data revealed several important patterns related to uranium mining operations in Jadugora.
          The findings suggest significant environmental impacts, with clear spatial and temporal patterns related to mining activities.
        </p>
      </div>
      
      <Tabs defaultValue="radiation" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="radiation">Radiation Levels</TabsTrigger>
          <TabsTrigger value="water">Water Quality</TabsTrigger>
          <TabsTrigger value="soil">Soil Contamination</TabsTrigger>
        </TabsList>
        
        <TabsContent value="radiation" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Radiation Levels</h2>
              <p className="mb-4">
                Radiation measurements showed significant spatial variation, with levels decreasing with distance from mining operations. 
                Temporal analysis indicated fluctuations corresponding to periods of increased mining activity.
              </p>
              <p className="mb-4">
                The mean radiation level in residential areas (0.79 μSv/h) was approximately 5 times higher than at control sites (0.15 μSv/h), 
                suggesting mining-related elevation.
              </p>
              <p>
                Our models identified tailings volume and ore extraction as the most influential predictors of radiation levels in surrounding areas.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/radiation_analysis.png" 
                alt="Radiation levels by distance from mining operations" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="water" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Water Quality</h2>
              <p className="mb-4">
                Water quality parameters showed concerning trends in areas downstream from mining operations. Uranium concentration 
                in water samples averaged 15.8 ppb, with some samples exceeding 30 ppb (the WHO guideline value).
              </p>
              <p className="mb-4">
                Heavy metals concentrations were also elevated in proximity to tailings ponds, with significant seasonal variation 
                corresponding to rainfall patterns.
              </p>
              <p>
                Water usage in mining operations and tailings volume were identified as the most important predictors of water contamination levels.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/water_quality_analysis.png" 
                alt="Water quality parameters by sampling location" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="soil" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Soil Contamination</h2>
              <p className="mb-4">
                Soil samples showed elevated levels of uranium, radium, lead, and arsenic in areas surrounding mining operations 
                and tailings disposal sites. Concentration gradients were observed, with levels decreasing with distance from these sites.
              </p>
              <p className="mb-4">
                Temporal analysis suggested accumulation of contaminants over time in certain areas, particularly those downwind from tailings.
              </p>
              <p>
                Distance from tailings disposal sites and cumulative waste generation were the most influential predictors of soil contamination levels.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/soil_contamination_analysis.png" 
                alt="Soil contamination levels by distance from tailings" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Environmental Impact Models</h2>
          <p className="mb-4">
            We developed models to predict environmental impacts based on mining activities. These models help quantify the relationships 
            between mining operations and environmental contamination, providing insights for mitigation strategies.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Radiation Level Prediction</h3>
              <p className="text-sm">
                The Lasso regression model performed best with an RMSE of 0.0325 μSv/h, though the negative R² value (-5.7076) indicates 
                challenges in capturing the complex relationships.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Water Uranium Prediction</h3>
              <p className="text-sm">
                The Gradient Boosting model performed best with an RMSE of 0.7523 ppb, though still with a negative R² value (-0.4412).
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-semibold mb-2">Soil Uranium Prediction</h3>
              <p className="text-sm">
                The Lasso model performed best with an RMSE of 0.4848 ppm and an R² of -1.0718.
              </p>
            </div>
          </div>
          
          <p>
            The negative R² values in environmental models suggest that the relationships between mining activities and environmental 
            contamination are complex and potentially non-linear. Additional data and more sophisticated modeling approaches may be 
            needed to better capture these relationships.
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Future Projections</h2>
          <p className="mb-4">
            Projections for environmental impacts suggest continued increases in contamination levels if mitigation measures are not strengthened. 
            The Ridge regression model performed best for radiation level projections, indicating a potential 15% increase in residential area 
            radiation levels over the next decade under current practices.
          </p>
          <div className="flex items-center justify-center">
            <img 
              src="/images/projection_radiation_level.png" 
              alt="Projected radiation levels over the next decade" 
              className="max-w-full h-auto rounded-lg shadow-md"
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default EnvironmentalImpact;
