import React from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';

const SocioeconomicImpact: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Socioeconomic Impact</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          Socioeconomic data analysis revealed complex impacts of mining operations in Jadugora, with both positive and negative effects
          on local communities. These impacts include changes in employment, education, and community displacement.
        </p>
      </div>
      
      <Tabs defaultValue="employment" className="w-full mb-8">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="employment">Employment</TabsTrigger>
          <TabsTrigger value="education">Education</TabsTrigger>
          <TabsTrigger value="displacement">Displacement</TabsTrigger>
        </TabsList>
        
        <TabsContent value="employment" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Employment</h2>
              <p className="mb-4">
                Mining operations provided direct employment for approximately 5,000 workers and indirect employment for an estimated 
                15,000 more in the region. However, employment benefits were unevenly distributed, with many technical positions filled 
                by workers from outside the local communities.
              </p>
              <p className="mb-4">
                Local employment was predominantly in lower-skilled positions with higher exposure risks, raising concerns about 
                occupational health and safety.
              </p>
              <p>
                Our analysis suggests that while mining operations contribute significantly to regional employment, more targeted 
                efforts are needed to ensure that local communities benefit equitably and safely from these opportunities.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/employment_analysis.png" 
                alt="Employment distribution by skill level and community origin" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="education" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Education</h2>
              <p className="mb-4">
                Educational outcomes showed mixed patterns. Communities with higher proportions of mining employees had improved 
                school infrastructure and higher enrollment rates, reflecting the economic benefits of mining employment.
              </p>
              <p className="mb-4">
                However, educational attainment was negatively correlated with proximity to mining operations, potentially due to 
                health impacts and socioeconomic disruption in these communities.
              </p>
              <p>
                These findings highlight the complex relationship between mining operations and educational outcomes, suggesting 
                the need for targeted educational support in affected communities.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/education_analysis.png" 
                alt="Educational attainment by community type" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
        
        <TabsContent value="displacement" className="p-4 border rounded-md mt-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h2 className="text-2xl font-semibold mb-4">Displacement</h2>
              <p className="mb-4">
                Mining operations led to displacement of approximately 7,500 people over the study period. Displaced communities 
                showed higher rates of poverty and unemployment compared to non-displaced communities.
              </p>
              <p className="mb-4">
                Compensation and rehabilitation measures showed limited effectiveness in restoring pre-displacement socioeconomic status, 
                with many displaced families experiencing long-term economic hardship.
              </p>
              <p>
                These findings underscore the importance of more effective compensation, rehabilitation, and livelihood restoration 
                programs for communities affected by mining-related displacement.
              </p>
            </div>
            <div className="flex items-center justify-center">
              <img 
                src="/images/displacement_analysis.png" 
                alt="Socioeconomic indicators for displaced vs. non-displaced communities" 
                className="max-w-full h-auto rounded-lg shadow-md"
              />
            </div>
          </div>
        </TabsContent>
      </Tabs>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Economic Development</h2>
          <p className="mb-4">
            Mining operations contributed to regional economic development through infrastructure improvements and supply chain activities.
            The economic multiplier effect of mining operations was estimated at 1.8, meaning that each job in mining supported an 
            additional 0.8 jobs in the regional economy.
          </p>
          <p className="mb-4">
            However, economic benefits were not evenly distributed, with communities closer to mining operations often experiencing 
            fewer benefits while bearing a disproportionate share of environmental and health impacts.
          </p>
          <p>
            Our analysis suggests that more inclusive economic development strategies are needed to ensure that all communities in 
            the region benefit from mining-related economic activities.
          </p>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Social Disruption</h2>
          <p className="mb-4">
            Communities near mining operations experienced various forms of social disruption, including changes in traditional 
            livelihoods and community structures. Traditional agricultural and forest-based livelihoods were particularly affected, 
            with many families forced to transition to wage labor or informal sector activities.
          </p>
          <p className="mb-4">
            Cultural impacts were also significant, with displacement and environmental changes affecting traditional practices 
            and community cohesion. Indigenous communities were particularly vulnerable to these changes.
          </p>
          <p>
            These findings highlight the importance of social impact assessment and mitigation strategies that address not only 
            economic impacts but also social and cultural dimensions of mining operations.
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default SocioeconomicImpact;
