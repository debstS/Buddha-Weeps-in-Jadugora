import React, { useState } from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Slider } from '../components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Label } from '../components/ui/label';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ZAxis } from 'recharts';

const Models: React.FC = () => {
  // State for interactive model parameters
  const [radiationDistance, setRadiationDistance] = useState(5);
  const [waterContamination, setWaterContamination] = useState(15);
  const [selectedYear, setSelectedYear] = useState('2025');
  const [selectedModel, setSelectedModel] = useState('random_forest');
  
  // Sample data for charts
  const modelPerformanceData = [
    { name: 'Linear Regression', cancer: 22.21, respiratory: 22.21, skin: 4.59, kidney: 10.60 },
    { name: 'Ridge', cancer: 15.65, respiratory: 15.65, skin: 3.56, kidney: 7.95 },
    { name: 'Lasso', cancer: 11.87, respiratory: 11.87, skin: 3.01, kidney: 3.24 },
    { name: 'Random Forest', cancer: 10.58, respiratory: 10.58, skin: 3.20, kidney: 5.45 },
    { name: 'Gradient Boosting', cancer: 11.13, respiratory: 11.13, skin: 2.00, kidney: 5.83 },
  ];
  
  const featureImportanceData = {
    cancer: [
      { name: 'Radiation Level', value: 0.32 },
      { name: 'Water Uranium', value: 0.28 },
      { name: 'Soil Uranium', value: 0.18 },
      { name: 'Distance from Mine', value: 0.12 },
      { name: 'Years of Operation', value: 0.10 },
    ],
    respiratory: [
      { name: 'Dust Generation', value: 0.35 },
      { name: 'Heavy Metals', value: 0.25 },
      { name: 'Radiation Level', value: 0.20 },
      { name: 'Distance from Mine', value: 0.15 },
      { name: 'Years of Operation', value: 0.05 },
    ],
    environmental: [
      { name: 'Tailings Volume', value: 0.30 },
      { name: 'Ore Extraction', value: 0.25 },
      { name: 'Water Usage', value: 0.20 },
      { name: 'Waste Generation', value: 0.15 },
      { name: 'Years of Operation', value: 0.10 },
    ]
  };
  
  const projectionData = [
    { year: 2020, actual: 22.5, projected: 22.5 },
    { year: 2021, actual: 23.2, projected: 23.1 },
    { year: 2022, actual: 24.1, projected: 23.8 },
    { year: 2023, actual: 24.8, projected: 24.5 },
    { year: 2024, actual: 25.5, projected: 25.2 },
    { year: 2025, actual: null, projected: 26.0 },
    { year: 2026, actual: null, projected: 26.8 },
    { year: 2027, actual: null, projected: 27.6 },
    { year: 2028, actual: null, projected: 28.5 },
    { year: 2029, actual: null, projected: 29.4 },
    { year: 2030, actual: null, projected: 30.3 },
  ];
  
  // Sample data for interactive prediction
  const calculatePrediction = () => {
    // This would normally call a real model API
    // Here we're using a simple formula for demonstration
    let basePrediction = 0;
    
    if (selectedModel === 'linear_regression') {
      basePrediction = 22.5;
    } else if (selectedModel === 'ridge') {
      basePrediction = 21.8;
    } else if (selectedModel === 'lasso') {
      basePrediction = 21.2;
    } else if (selectedModel === 'random_forest') {
      basePrediction = 20.5;
    } else if (selectedModel === 'gradient_boosting') {
      basePrediction = 20.8;
    }
    
    // Adjust based on parameters
    const distanceFactor = 1 - (radiationDistance / 20) * 0.5; // Further distance = lower prediction
    const contaminationFactor = 1 + (waterContamination / 30) * 0.8; // Higher contamination = higher prediction
    const yearFactor = 1 + ((parseInt(selectedYear) - 2025) / 10) * 0.2; // Later year = higher prediction
    
    return (basePrediction * distanceFactor * contaminationFactor * yearFactor).toFixed(1);
  };
  
  // Scatter plot data for relationship visualization
  const relationshipData = Array.from({ length: 50 }, (_, _i) => ({
    distance: 1 + Math.random() * 19,
    contamination: 5 + Math.random() * 25,
    rate: 15 + Math.random() * 20,
  }));

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Predictive Models</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          We developed machine learning models to predict environmental contamination levels, health outcomes, and future impacts 
          based on mining activities. These models help quantify relationships and provide insights for policy decisions and 
          mitigation strategies.
        </p>
      </div>
      
      <Tabs defaultValue="performance" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
          <TabsTrigger value="features">Feature Importance</TabsTrigger>
          <TabsTrigger value="projections">Future Projections</TabsTrigger>
          <TabsTrigger value="interactive">Interactive Prediction</TabsTrigger>
        </TabsList>
        
        <TabsContent value="performance" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Model Performance Comparison</h2>
          <p className="mb-4">
            This chart compares the Root Mean Square Error (RMSE) of different models for predicting health outcomes.
            Lower values indicate better performance.
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={modelPerformanceData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'RMSE (cases per 10,000)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="cancer" name="Cancer Rate" fill="#8884d8" />
                <Bar dataKey="respiratory" name="Respiratory Disease Rate" fill="#82ca9d" />
                <Bar dataKey="skin" name="Skin Disorders Rate" fill="#ffc658" />
                <Bar dataKey="kidney" name="Kidney Disease Rate" fill="#ff8042" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            Note: Lower RMSE values indicate better model performance. The Gradient Boosting model performed best for 
            skin disorders prediction, while Random Forest models performed best for cancer and respiratory disease prediction.
          </p>
        </TabsContent>
        
        <TabsContent value="features" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Feature Importance Analysis</h2>
          <p className="mb-4">
            This chart shows the relative importance of different features in our predictive models.
          </p>
          <Tabs defaultValue="cancer">
            <TabsList>
              <TabsTrigger value="cancer">Cancer Rate Model</TabsTrigger>
              <TabsTrigger value="respiratory">Respiratory Disease Model</TabsTrigger>
              <TabsTrigger value="environmental">Environmental Model</TabsTrigger>
            </TabsList>
            <TabsContent value="cancer" className="mt-4">
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={featureImportanceData.cancer}
                    margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 0.4]} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" name="Importance Score" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            <TabsContent value="respiratory" className="mt-4">
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={featureImportanceData.respiratory}
                    margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 0.4]} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" name="Importance Score" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
            <TabsContent value="environmental" className="mt-4">
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    layout="vertical"
                    data={featureImportanceData.environmental}
                    margin={{ top: 20, right: 30, left: 100, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[0, 0.4]} />
                    <YAxis type="category" dataKey="name" />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="value" name="Importance Score" fill="#ffc658" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </TabsContent>
        
        <TabsContent value="projections" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Future Projections</h2>
          <p className="mb-4">
            This chart shows actual and projected cancer rates over time, based on our predictive models.
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={projectionData}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: 'Cancer Rate (per 10,000)', angle: -90, position: 'insideLeft' }} domain={[20, 35]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" name="Actual Rate" stroke="#8884d8" activeDot={{ r: 8 }} />
                <Line type="monotone" dataKey="projected" name="Projected Rate" stroke="#82ca9d" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            Note: Projections suggest a potential 18% increase in cancer rates over the next decade under current conditions.
            These projections should be interpreted with caution given the limitations of the models and the potential for
            policy interventions or technological improvements to alter these trajectories.
          </p>
        </TabsContent>
        
        <TabsContent value="interactive" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Interactive Prediction Tool</h2>
          <p className="mb-4">
            Adjust the parameters below to see how different factors affect predicted cancer rates in communities near mining operations.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium mb-4">Model Parameters</h3>
                
                <div className="space-y-6">
                  <div>
                    <Label htmlFor="model-select" className="mb-2 block">Select Model</Label>
                    <Select value={selectedModel} onValueChange={setSelectedModel}>
                      <SelectTrigger id="model-select">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="linear_regression">Linear Regression</SelectItem>
                        <SelectItem value="ridge">Ridge Regression</SelectItem>
                        <SelectItem value="lasso">Lasso Regression</SelectItem>
                        <SelectItem value="random_forest">Random Forest</SelectItem>
                        <SelectItem value="gradient_boosting">Gradient Boosting</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div>
                    <Label htmlFor="distance-slider" className="mb-2 block">
                      Distance from Mining Operations (km): {radiationDistance}
                    </Label>
                    <Slider
                      id="distance-slider"
                      min={1}
                      max={20}
                      step={1}
                      value={[radiationDistance]}
                      onValueChange={(value) => setRadiationDistance(value[0])}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="contamination-slider" className="mb-2 block">
                      Water Uranium Concentration (ppb): {waterContamination}
                    </Label>
                    <Slider
                      id="contamination-slider"
                      min={5}
                      max={30}
                      step={1}
                      value={[waterContamination]}
                      onValueChange={(value) => setWaterContamination(value[0])}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="year-select" className="mb-2 block">Projection Year</Label>
                    <Select value={selectedYear} onValueChange={setSelectedYear}>
                      <SelectTrigger id="year-select">
                        <SelectValue placeholder="Select year" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2025">2025</SelectItem>
                        <SelectItem value="2026">2026</SelectItem>
                        <SelectItem value="2027">2027</SelectItem>
                        <SelectItem value="2028">2028</SelectItem>
                        <SelectItem value="2029">2029</SelectItem>
                        <SelectItem value="2030">2030</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-6">
                <h3 className="text-lg font-medium mb-4">Prediction Results</h3>
                
                <div className="flex flex-col items-center justify-center h-full">
                  <div className="text-6xl font-bold text-primary mb-4">
                    {calculatePrediction()}
                  </div>
                  <div className="text-lg text-center">
                    Predicted cancer rate per 10,000 population
                  </div>
                  <div className="text-sm text-muted-foreground text-center mt-4">
                    Based on {selec
(Content truncated due to size limit. Use line ranges to read in chunks)