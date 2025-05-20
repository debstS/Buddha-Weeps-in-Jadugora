import React from 'react';
import { Card, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Link } from 'react-router-dom';
import { AlertTriangle } from 'lucide-react';

const Recommendations: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8 text-center">Recommendations</h1>
      
      <div className="mb-8">
        <p className="text-lg mb-4">
          Based on our comprehensive analysis of the environmental, health, and socioeconomic impacts of uranium mining in Jadugora,
          we propose the following recommendations to address identified issues and improve outcomes for affected communities.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <Card>
          <CardContent className="pt-6">
            <h2 className="text-2xl font-semibold mb-4 flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-yellow-500" />
              Policy Recommendations
            </h2>
            <ul className="space-y-4">
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">1</div>
                <div>
                  <h3 className="font-medium">Enhanced Environmental Monitoring</h3>
                  <p className="text-sm text-muted-foreground">
                    Implement comprehensive monitoring programs for radiation, water quality, and soil contamination, with public reporting of results.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">2</div>
                <div>
                  <h3 className="font-medium">Health Surveillance</h3>
                  <p className="text-sm text-muted-foreground">
                    Establish systematic health surveillance in affected communities, with particular attention to conditions potentially related to radiation and heavy metal exposure.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">3</div>
                <div>
                  <h3 className="font-medium">Regulatory Framework</h3>
                  <p className="text-sm text-muted-foreground">
                    Strengthen regulatory standards and enforcement for uranium mining operations, particularly regarding waste management and tailings disposal.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">4</div>
                <div>
                  <h3 className="font-medium">Community Involvement</h3>
                  <p className="text-sm text-muted-foreground">
                    Ensure meaningful participation of affected communities in decision-making processes regarding mining operations and mitigation measures.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">5</div>
                <div>
                  <h3 className="font-medium">Compensation and Rehabilitation</h3>
                  <p className="text-sm text-muted-foreground">
                    Develop more effective compensation and rehabilitation programs for displaced communities and those experiencing health impacts.
                  </p>
                </div>
              </li>
            </ul>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-6">
            <h2 className="text-2xl font-semibold mb-4 flex items-center">
              <AlertTriangle className="mr-2 h-5 w-5 text-blue-500" />
              Mitigation Strategies
            </h2>
            <ul className="space-y-4">
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">1</div>
                <div>
                  <h3 className="font-medium">Improved Tailings Management</h3>
                  <p className="text-sm text-muted-foreground">
                    Implement best practices for tailings disposal, including lined storage facilities, cover systems to reduce dust, and enhanced monitoring.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">2</div>
                <div>
                  <h3 className="font-medium">Water Treatment</h3>
                  <p className="text-sm text-muted-foreground">
                    Enhance water treatment processes for mining effluents and implement groundwater remediation in affected areas.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">3</div>
                <div>
                  <h3 className="font-medium">Dust Control</h3>
                  <p className="text-sm text-muted-foreground">
                    Strengthen dust suppression measures at mining sites and transportation routes to reduce airborne contamination.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">4</div>
                <div>
                  <h3 className="font-medium">Clean Water Supply</h3>
                  <p className="text-sm text-muted-foreground">
                    Provide alternative water supplies for communities with contaminated water sources.
                  </p>
                </div>
              </li>
              <li className="flex">
                <div className="mr-4 mt-1 bg-primary text-primary-foreground rounded-full h-6 w-6 flex items-center justify-center text-sm">5</div>
                <div>
                  <h3 className="font-medium">Occupational Safety</h3>
                  <p className="text-sm text-muted-foreground">
                    Enhance radiation protection measures for workers, including improved ventilation, personal protective equipment, and exposure monitoring.
                  </p>
                </div>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
      
      <Card className="mb-8">
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4 flex items-center">
            <AlertTriangle className="mr-2 h-5 w-5 text-green-500" />
            Future Research Directions
          </h2>
          <p className="mb-4">
            We recommend the following areas for future research to address current knowledge gaps and improve impact assessment and mitigation strategies:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Longitudinal Health Studies</h3>
              <p className="text-sm">
                Conduct long-term health studies in affected communities to better understand exposure-outcome relationships and track changes over time.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Biomonitoring</h3>
              <p className="text-sm">
                Implement biomonitoring programs to measure actual body burden of contaminants in exposed populations and establish dose-response relationships.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Remediation Technologies</h3>
              <p className="text-sm">
                Research and develop cost-effective remediation technologies for contaminated soil and water in the specific context of uranium mining areas.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Improved Modeling Approaches</h3>
              <p className="text-sm">
                Develop more sophisticated modeling approaches that can better capture complex environmental and health relationships in mining-affected areas.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Socioeconomic Interventions</h3>
              <p className="text-sm">
                Evaluate the effectiveness of various socioeconomic interventions in mitigating the negative impacts of mining operations on local communities.
              </p>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Community-Based Monitoring</h3>
              <p className="text-sm">
                Develop and evaluate community-based environmental and health monitoring programs that empower local communities to participate in data collection and analysis.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardContent className="pt-6">
          <h2 className="text-2xl font-semibold mb-4">Implementation Roadmap</h2>
          <p className="mb-4">
            We propose a phased approach to implementing these recommendations:
          </p>
          <div className="space-y-4">
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Phase 1: Immediate Actions (0-12 months)</h3>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Establish comprehensive environmental monitoring program</li>
                <li>Provide alternative water supplies to most affected communities</li>
                <li>Implement enhanced dust control measures</li>
                <li>Begin community health screening program</li>
              </ul>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Phase 2: Medium-Term Actions (1-3 years)</h3>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Upgrade tailings management facilities</li>
                <li>Implement water treatment improvements</li>
                <li>Develop and implement enhanced compensation programs</li>
                <li>Establish long-term health surveillance system</li>
              </ul>
            </div>
            <div className="border p-4 rounded-md">
              <h3 className="font-medium mb-2">Phase 3: Long-Term Actions (3-5 years)</h3>
              <ul className="list-disc pl-6 space-y-1 text-sm">
                <li>Implement comprehensive remediation of contaminated areas</li>
                <li>Develop sustainable economic alternatives for affected communities</li>
                <li>Establish permanent monitoring and evaluation system</li>
                <li>Update regulatory framework based on research findings</li>
              </ul>
            </div>
          </div>
          <div className="mt-6 text-center">
            <Button asChild>
              <Link to="/models">Explore our predictive models</Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Recommendations;
