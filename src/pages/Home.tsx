import React from 'react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { ArrowRight, FileText, AlertTriangle } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col items-center text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          Jadugora Uranium Mining Impact Assessment
        </h1>
        <p className="text-xl text-muted-foreground max-w-3xl">
          A comprehensive data science project analyzing the environmental, health, and societal consequences 
          of uranium mining in Jadugora, Jharkhand, India.
        </p>
        <div className="flex gap-4 mt-8">
          <Button asChild>
            <Link to="/findings">
              Key Findings <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button variant="outline" asChild>
            <Link to="/methodology">
              Methodology <FileText className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-yellow-500" />
              Environmental Impact
            </CardTitle>
            <CardDescription>Analysis of radiation, water and soil contamination</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Our analysis revealed significant environmental impacts, with radiation levels in residential areas 
            approximately 5 times higher than control sites and concerning levels of uranium in water samples.</p>
          </CardContent>
          <CardFooter>
            <Button variant="ghost" asChild>
              <Link to="/environmental-impact">Learn more <ArrowRight className="ml-2 h-4 w-4" /></Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-red-500" />
              Health Impact
            </CardTitle>
            <CardDescription>Analysis of disease prevalence and health outcomes</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Communities closer to mining operations showed higher rates of certain diseases, with cancer rates 
            approximately twice as high as more distant communities and elevated rates of respiratory diseases.</p>
          </CardContent>
          <CardFooter>
            <Button variant="ghost" asChild>
              <Link to="/health-impact">Learn more <ArrowRight className="ml-2 h-4 w-4" /></Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-blue-500" />
              Socioeconomic Impact
            </CardTitle>
            <CardDescription>Analysis of employment, education, and displacement</CardDescription>
          </CardHeader>
          <CardContent>
            <p>Mining operations provided significant employment opportunities but with uneven distribution of benefits. 
            Approximately 7,500 people were displaced, with these communities showing higher rates of poverty.</p>
          </CardContent>
          <CardFooter>
            <Button variant="ghost" asChild>
              <Link to="/socioeconomic-impact">Learn more <ArrowRight className="ml-2 h-4 w-4" /></Link>
            </Button>
          </CardFooter>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="mb-12">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="mlops">MLOps</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="p-4 border rounded-md mt-2">
          <h3 className="text-xl font-semibold mb-2">Project Overview</h3>
          <p className="mb-4">
            This data science project provides a comprehensive assessment of the environmental, health, and socioeconomic 
            impacts of uranium mining in Jadugora, Jharkhand, India. Using advanced data analysis and machine learning 
            techniques, we have developed models to quantify relationships between mining activities and various impact indicators.
          </p>
          <Button variant="outline" asChild>
            <Link to="/about">Read full project background</Link>
          </Button>
        </TabsContent>
        <TabsContent value="data" className="p-4 border rounded-md mt-2">
          <h3 className="text-xl font-semibold mb-2">Data Collection & Analysis</h3>
          <p className="mb-4">
            We collected and analyzed data on environmental parameters (radiation, water quality, soil contamination), 
            health outcomes (disease prevalence, birth defects), socioeconomic factors (employment, education), and 
            mining production metrics (ore extraction, waste generation).
          </p>
          <Button variant="outline" asChild>
            <Link to="/data">Explore the data</Link>
          </Button>
        </TabsContent>
        <TabsContent value="models" className="p-4 border rounded-md mt-2">
          <h3 className="text-xl font-semibold mb-2">Predictive Models</h3>
          <p className="mb-4">
            We developed machine learning models to predict environmental contamination levels, health outcomes, and 
            future impacts based on mining activities. These models help quantify relationships and provide insights 
            for policy decisions and mitigation strategies.
          </p>
          <Button variant="outline" asChild>
            <Link to="/models">Explore the models</Link>
          </Button>
        </TabsContent>
        <TabsContent value="mlops" className="p-4 border rounded-md mt-2">
          <h3 className="text-xl font-semibold mb-2">MLOps Implementation</h3>
          <p className="mb-4">
            We implemented a comprehensive MLOps pipeline to automate the training, evaluation, and deployment of our 
            impact assessment models, enabling ongoing monitoring and updates as new data becomes available.
          </p>
          <Button variant="outline" asChild>
            <Link to="/mlops">Learn about our MLOps approach</Link>
          </Button>
        </TabsContent>
      </Tabs>

      <div className="bg-muted p-6 rounded-lg">
        <h2 className="text-2xl font-bold mb-4">Key Recommendations</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-start">
            <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full p-1">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-medium">Enhanced Environmental Monitoring</h3>
              <p className="text-sm text-muted-foreground">
                Implement comprehensive monitoring programs for radiation, water quality, and soil contamination, 
                with public reporting of results.
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full p-1">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-medium">Health Surveillance</h3>
              <p className="text-sm text-muted-foreground">
                Establish systematic health surveillance in affected communities, with particular attention to 
                conditions potentially related to radiation and heavy metal exposure.
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full p-1">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-medium">Improved Tailings Management</h3>
              <p className="text-sm text-muted-foreground">
                Implement best practices for tailings disposal, including lined storage facilities, cover systems 
                to reduce dust, and enhanced monitoring.
              </p>
            </div>
          </div>
          <div className="flex items-start">
            <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full p-1">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="font-medium">Community Involvement</h3>
              <p className="text-sm text-muted-foreground">
                Ensure meaningful participation of affected communities in decision-making processes regarding 
                mining operations and mitigation measures.
              </p>
            </div>
          </div>
        </div>
        <div className="mt-4 text-center">
          <Button asChild>
            <Link to="/recommendations">View all recommendations</Link>
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Home;
