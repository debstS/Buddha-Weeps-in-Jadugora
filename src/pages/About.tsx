import React from 'react';
import { Card, CardContent } from '../components/ui/card';

const About: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">About the Project</h1>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Background</h2>
          <p className="mb-4">
            Uranium mining has been conducted in Jadugora, Jharkhand, India since the 1960s, providing essential fuel for India's nuclear power program. 
            The Uranium Corporation of India Limited (UCIL) operates these mines, which are among the oldest uranium mining operations in the country. 
            While these operations contribute significantly to India's energy security and economic development, concerns have been raised about their 
            environmental and health impacts on local communities.
          </p>
          <p className="mb-4">
            The documentary "Buddha Weeps in Jadugora" (1999) by filmmaker Shriprakash brought international attention to these issues, highlighting 
            reports of elevated radiation levels, contaminated water sources, and increased incidence of health problems in communities surrounding the mines. 
            Despite the controversy, comprehensive scientific studies examining the relationships between mining activities and various impact indicators 
            have been limited.
          </p>
          <p>
            This project aims to address this gap by applying data science and machine learning techniques to analyze available data and develop predictive 
            models that can help understand and quantify these impacts. By creating a robust analytical framework, we seek to provide evidence-based insights 
            that can inform policy decisions and mitigation strategies.
          </p>
        </CardContent>
      </Card>

      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Project Objectives</h2>
          <p className="mb-4">The primary objectives of this data science project are:</p>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>To collect, process, and analyze data related to uranium mining operations in Jadugora and their potential environmental, health, and socioeconomic impacts</li>
            <li>To develop machine learning models that can quantify relationships between mining activities and various impact indicators</li>
            <li>To create predictive models for assessing future environmental and health impacts under different scenarios</li>
            <li>To implement a cloud-based MLOps pipeline for model deployment, monitoring, and maintenance</li>
            <li>To provide data-driven recommendations for policy interventions and mitigation strategies</li>
          </ul>
        </CardContent>
      </Card>

      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Scope and Limitations</h2>
          <p className="mb-4">This project encompasses the following scope:</p>
          <ul className="list-disc pl-6 space-y-2 mb-4">
            <li>Analysis of environmental data including radiation levels, water quality, and soil contamination in the Jadugora region</li>
            <li>Assessment of health data including disease prevalence, birth defects, and mortality rates in communities near mining operations</li>
            <li>Examination of socioeconomic factors including employment, education, and displacement</li>
            <li>Development of predictive models for environmental contamination and health outcomes</li>
            <li>Implementation of a complete MLOps pipeline for model deployment and monitoring</li>
          </ul>
          
          <p className="mb-4">Key limitations of this study include:</p>
          <ul className="list-disc pl-6 space-y-2">
            <li>Reliance on synthetic data that simulates real-world patterns due to limited availability of comprehensive field measurements</li>
            <li>Challenges in establishing causality versus correlation in observed relationships</li>
            <li>Limited temporal and spatial resolution in available datasets</li>
            <li>Potential confounding factors not captured in the available data</li>
            <li>Technical limitations in the model registration process for environmental impact models</li>
          </ul>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Project Team</h2>
          <p className="mb-4">
            This project was conducted by a multidisciplinary team of data scientists, environmental specialists, and health researchers, 
            with input from community representatives and policy experts. The team employed a collaborative approach to ensure that the 
            analysis addressed the complex interrelationships between mining activities and their various impacts.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default About;
