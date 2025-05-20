import React, { useState } from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Label } from '../components/ui/label';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

const Data: React.FC = () => {
  // State for interactive data exploration
  const [selectedDataset, setSelectedDataset] = useState('environmental');
  const [selectedYear, setSelectedYear] = useState('2023');
  const [selectedLocation, setSelectedLocation] = useState('high_exposure');
  
  // Sample data for charts
  const environmentalData = {
    radiation: [
      { year: 2018, high_exposure: 0.65, medium_exposure: 0.42, low_exposure: 0.22, control: 0.15 },
      { year: 2019, high_exposure: 0.68, medium_exposure: 0.45, low_exposure: 0.23, control: 0.15 },
      { year: 2020, high_exposure: 0.72, medium_exposure: 0.47, low_exposure: 0.24, control: 0.15 },
      { year: 2021, high_exposure: 0.75, medium_exposure: 0.49, low_exposure: 0.24, control: 0.15 },
      { year: 2022, high_exposure: 0.77, medium_exposure: 0.50, low_exposure: 0.25, control: 0.15 },
      { year: 2023, high_exposure: 0.79, medium_exposure: 0.51, low_exposure: 0.25, control: 0.15 },
    ],
    water: [
      { year: 2018, high_exposure: 12.5, medium_exposure: 8.2, low_exposure: 4.1, control: 1.2 },
      { year: 2019, high_exposure: 13.2, medium_exposure: 8.6, low_exposure: 4.3, control: 1.2 },
      { year: 2020, high_exposure: 14.1, medium_exposure: 9.2, low_exposure: 4.6, control: 1.3 },
      { year: 2021, high_exposure: 14.8, medium_exposure: 9.6, low_exposure: 4.8, control: 1.3 },
      { year: 2022, high_exposure: 15.3, medium_exposure: 10.0, low_exposure: 5.0, control: 1.3 },
      { year: 2023, high_exposure: 15.8, medium_exposure: 10.3, low_exposure: 5.2, control: 1.4 },
    ],
    soil: [
      { year: 2018, high_exposure: 3.2, medium_exposure: 2.1, low_exposure: 1.0, control: 0.3 },
      { year: 2019, high_exposure: 3.4, medium_exposure: 2.2, low_exposure: 1.1, control: 0.3 },
      { year: 2020, high_exposure: 3.6, medium_exposure: 2.3, low_exposure: 1.2, control: 0.3 },
      { year: 2021, high_exposure: 3.8, medium_exposure: 2.5, low_exposure: 1.2, control: 0.3 },
      { year: 2022, high_exposure: 4.0, medium_exposure: 2.6, low_exposure: 1.3, control: 0.3 },
      { year: 2023, high_exposure: 4.2, medium_exposure: 2.7, low_exposure: 1.4, control: 0.3 },
    ]
  };
  
  const healthData = {
    cancer: [
      { year: 2018, high_exposure: 21.5, medium_exposure: 16.2, low_exposure: 11.8, control: 10.5 },
      { year: 2019, high_exposure: 22.3, medium_exposure: 16.8, low_exposure: 12.1, control: 10.6 },
      { year: 2020, high_exposure: 23.1, medium_exposure: 17.4, low_exposure: 12.4, control: 10.7 },
      { year: 2021, high_exposure: 23.9, medium_exposure: 18.0, low_exposure: 12.6, control: 10.8 },
      { year: 2022, high_exposure: 24.6, medium_exposure: 18.5, low_exposure: 12.7, control: 10.8 },
      { year: 2023, high_exposure: 25.2, medium_exposure: 19.0, low_exposure: 12.8, control: 10.9 },
    ],
    respiratory: [
      { year: 2018, high_exposure: 32.5, medium_exposure: 24.2, low_exposure: 18.8, control: 17.5 },
      { year: 2019, high_exposure: 33.8, medium_exposure: 25.1, low_exposure: 19.2, control: 17.6 },
      { year: 2020, high_exposure: 35.1, medium_exposure: 26.0, low_exposure: 19.6, control: 17.7 },
      { year: 2021, high_exposure: 36.4, medium_exposure: 26.9, low_exposure: 20.0, control: 17.8 },
      { year: 2022, high_exposure: 37.5, medium_exposure: 27.7, low_exposure: 20.3, control: 17.9 },
      { year: 2023, high_exposure: 38.5, medium_exposure: 28.4, low_exposure: 20.6, control: 18.0 },
    ],
    birth_defects: [
      { year: 2018, high_exposure: 8.2, medium_exposure: 6.1, low_exposure: 4.8, control: 4.5 },
      { year: 2019, high_exposure: 8.5, medium_exposure: 6.3, low_exposure: 4.9, control: 4.5 },
      { year: 2020, high_exposure: 8.8, medium_exposure: 6.5, low_exposure: 5.0, control: 4.6 },
      { year: 2021, high_exposure: 9.1, medium_exposure: 6.7, low_exposure: 5.1, control: 4.6 },
      { year: 2022, high_exposure: 9.4, medium_exposure: 6.9, low_exposure: 5.2, control: 4.7 },
      { year: 2023, high_exposure: 9.7, medium_exposure: 7.1, low_exposure: 5.3, control: 4.7 },
    ]
  };
  
  const socioeconomicData = {
    employment: [
      { year: 2018, direct: 4800, indirect: 14400, total: 19200 },
      { year: 2019, direct: 4850, indirect: 14550, total: 19400 },
      { year: 2020, direct: 4900, indirect: 14700, total: 19600 },
      { year: 2021, direct: 4950, indirect: 14850, total: 19800 },
      { year: 2022, direct: 5000, indirect: 15000, total: 20000 },
      { year: 2023, direct: 5050, indirect: 15150, total: 20200 },
    ],
    education: [
      { year: 2018, mining_communities: 68, other_communities: 72, displaced_communities: 62 },
      { year: 2019, mining_communities: 69, other_communities: 73, displaced_communities: 63 },
      { year: 2020, mining_communities: 70, other_communities: 74, displaced_communities: 64 },
      { year: 2021, mining_communities: 71, other_communities: 75, displaced_communities: 65 },
      { year: 2022, mining_communities: 72, other_communities: 76, displaced_communities: 66 },
      { year: 2023, mining_communities: 73, other_communities: 77, displaced_communities: 67 },
    ],
    displacement: [
      { year: 2018, count: 6800 },
      { year: 2019, count: 7000 },
      { year: 2020, count: 7200 },
      { year: 2021, count: 7350 },
      { year: 2022, count: 7450 },
      { year: 2023, count: 7500 },
    ]
  };
  
  const miningData = {
    production: [
      { year: 2018, ore_extraction: 85000, uranium_production: 42500 },
      { year: 2019, ore_extraction: 88000, uranium_production: 44000 },
      { year: 2020, ore_extraction: 92000, uranium_production: 46000 },
      { year: 2021, ore_extraction: 98000, uranium_production: 49000 },
      { year: 2022, ore_extraction: 105000, uranium_production: 52500 },
      { year: 2023, ore_extraction: 112000, uranium_production: 56000 },
    ],
    waste: [
      { year: 2018, waste_rock: 340000, tailings: 76500 },
      { year: 2019, waste_rock: 352000, tailings: 79200 },
      { year: 2020, waste_rock: 368000, tailings: 82800 },
      { year: 2021, waste_rock: 392000, tailings: 88200 },
      { year: 2022, waste_rock: 420000, tailings: 94500 },
      { year: 2023, waste_rock: 448000, tailings: 100800 },
    ],
    water: [
      { year: 2018, usage: 425 },
      { year: 2019, usage: 440 },
      { year: 2020, usage: 460 },
      { year: 2021, usage: 490 },
      { year: 2022, usage: 525 },
      { year: 2023, usage: 560 },
    ]
  };
  
  // Distribution data for pie charts
  const distributionData = {
    environmental: [
      { name: 'Radiation', value: 35 },
      { name: 'Water Contamination', value: 40 },
      { name: 'Soil Contamination', value: 25 },
    ],
    health: [
      { name: 'Cancer', value: 30 },
      { name: 'Respiratory', value: 35 },
      { name: 'Skin Disorders', value: 20 },
      { name: 'Kidney Disease', value: 15 },
    ],
    socioeconomic: [
      { name: 'Employment', value: 45 },
      { name: 'Education', value: 25 },
      { name: 'Displacement', value: 30 },
    ]
  };
  
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];
  
  // Helper function to get appropriate data based on selections
  const getChartData = () => {
    if (selectedDataset === 'environmental') {
      if (selectedYear === 'all') {
        return environmentalData.radiation;
      } else {
        return environmentalData.radiation.filter(item => item.year.toString() === selectedYear);
      }
    } else if (selectedDataset === 'health') {
      if (selectedYear === 'all') {
        return healthData.cancer;
      } else {
        return healthData.cancer.filter(item => item.year.toString() === selectedYear);
      }
    } else if (selectedDataset === 'socioeconomic') {
      if (selectedYear === 'all') {
        return socioeconomicData.employment;
      } else {
        return socioeconomicData.employment.filter(item => item.year.toString() === selectedYear);
      }
    } else if (selectedDataset === 'mining') {
      if (selectedYear === 'all') {
        return miningData.production;
      } else {
        return miningData.production.filter(item => item.year.toString() === selectedYear);
      }
    }
    return [];
  };
  
  // Helper function to get distribution data
  const getDistributionData = () => {
    if (selectedDataset === 'environmental') {
      return distributionData.environmental;
    } else if (selectedDataset === 'health') {
      return distributionData.health;
    } else if (selectedDataset === 'socioeconomic') {
      return distributionData.socioeconomic;
    }
    return [];
  };
  
  // Helper function to get trend data
  const getTrendData = () => {
    if (selectedDataset === 'environmental') {
      if (selectedLocation === 'high_exposure') {
        return environmentalData.radiation.map(item => ({
          year: item.year,
          value: item.high_exposure
        }));
      } else if (selectedLocation === 'medium_exposure') {
        return environmentalData.radiation.map(item => ({
          year: item.year,
          value: item.medium_exposure
        }));
      } else if (selectedLocation === 'low_exposure') {
        return environmentalData.radiation.map(item => ({
          year: item.year,
          value: item.low_exposure
        }));
      } else {
        return environmentalData.radiation.map(item => ({
          year: item.year,
          value: item.control
        }));
      }
    } else if (selectedDataset === 'health') {
      if (selectedLocation === 'high_exposure') {
        return healthData.cancer.map(item => ({
          year: item.year,
          value: item.high_exposure
        }));
      } else if (selectedLocation === 'medium_exposure') {
        return healthData.cancer.map(item => ({
          year: item.year,
          value: item.medium_exposure
        }));
      } else if (selectedLocation === 'low_exposure') {
        return healthData.cancer.map(item => ({
          year: item.year,
          value: item.low_exposure
        }));
      } else {
        return healthData.cancer.map(item => ({
          year: item.year,
          value: item.control
        }));
      }
    } else if (selectedDataset === 'socioeconomic') {
      return socioeconomicData.employment.map(item => ({
        year: item.year,
        value: item.total
      }));
    } else if (selectedDataset === 'mining') {
      return miningData.production.map(item => ({
        year: item.year,
        value: item.ore_extraction
      }));
    }
    return [];
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Data Explorer</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          Explore the data collected and analyzed in our study of uranium mining impacts in Jadugora. 
          Use the controls below to view different datasets, time periods, and visualizations.
        </p>
      </div>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Data Selection</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div>
              <Label htmlFor="dataset-select" className="mb-2 block">Dataset Category</Label>
              <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                <SelectTrigger id="dataset-select">
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="environmental">Environmental Data</SelectItem>
                  <SelectItem value="health">Health Data</SelectItem>
                  <SelectItem value="socioeconomic">Socioeconomic Data</SelectItem>
                  <SelectItem value="mining">Mining Production Data</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="year-select" className="mb-2 block">Year</Label>
              <Select value={selectedYear} onValueChange={setSelectedYear}>
                <SelectTrigger id="year-select">
                  <SelectValue placeholder="Select year" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Years</SelectItem>
                  <SelectItem value="2018">2018</SelectItem>
                  <SelectItem value="2019">2019</SelectItem>
                  <SelectItem value="2020">2020</SelectItem>
                  <SelectItem value="2021">2021</SelectItem>
                  <SelectItem value="2022">2022</SelectItem>
                  <SelectItem value="2023">2023</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div>
              <Label htmlFor="location-select" className="mb-2 block">Location/Exposure Level</Label>
              <Select value={selectedLocation} onValueChange={setSelectedLocation}>
                <SelectTrigger id="location-select">
                  <SelectValue placeholder="Select location" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="high_exposure">High Exposure Areas</SelectItem>
                  <SelectItem value="medium_exposure">Medium Exposure Areas</SelectItem>
                  <SelectItem value="low_exposure">Low Exposure Areas</SelectItem>
                  <SelectItem value="control">Control Areas</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Tabs defaultValue="trends" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="trends">Trends Over Time</TabsTrigger>
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
        </TabsList>
        
        <TabsContent value="trends" className="p-4 border rounded-md mt-2">
          <h2 className="text-2xl font-semibold mb-4">Trends Over Time</h2>
          <p className="mb-4">
            This chart shows how key indicators have changed over the study period (2018-2023).
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={getTrendData()}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="value" name={`${selectedDataset.charAt(0).toUpperCase() + selectedDataset.slice(1)} Indicator`} stroke="#8884d8" activeDot={{ r: 8 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-muted-foreground">
            Note: The y-axis units vary depending on the selected dataset. For environmental data, units are μSv/h (radiation), 
            ppb (water uranium), or ppm (soil uranium). For health data, units are cases per 10,000 population.
          </p>
        </TabsContent>
        
        <TabsContent value="comparison" className="p-4 border rounded-md mt-2">
          <h2 className="tex
(Content truncated due to size limit. Use line ranges to read in chunks)