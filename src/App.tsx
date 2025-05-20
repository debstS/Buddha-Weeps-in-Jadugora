import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { ThemeProvider } from './components/ui/theme-provider';
// Button import removed as it's not used
import { 
  Home, 
  FileText, 
  BarChart2, 
  Database, 
  AlertTriangle, 
  Users, 
  BookOpen, 
  Menu, 
  X 
} from 'lucide-react';

// Import pages
import HomePage from './pages/Home';
import AboutPage from './pages/About';
import MethodologyPage from './pages/Methodology';
import EnvironmentalImpactPage from './pages/EnvironmentalImpact';
import HealthImpactPage from './pages/HealthImpact';
import SocioeconomicImpactPage from './pages/SocioeconomicImpact';
import RecommendationsPage from './pages/Recommendations';
import ModelsPage from './pages/Models';
import DataPage from './pages/Data';

function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setMobileMenuOpen(false);
  };

  return (
    <ThemeProvider defaultTheme="light" storageKey="jadugora-theme">
      <Router>
        <div className="min-h-screen flex flex-col">
          {/* Header */}
          <header className="bg-primary text-primary-foreground py-4 px-6 shadow-md">
            <div className="container mx-auto flex justify-between items-center">
              <Link to="/" className="text-xl font-bold flex items-center" onClick={closeMobileMenu}>
                <AlertTriangle className="mr-2 h-6 w-6" />
                <span className="hidden sm:inline">Jadugora Impact Assessment</span>
                <span className="sm:hidden">JIA</span>
              </Link>
              
              {/* Desktop Navigation */}
              <nav className="hidden md:flex space-x-4">
                <Link to="/" className="hover:underline flex items-center">
                  <Home className="mr-1 h-4 w-4" />
                  Home
                </Link>
                <Link to="/about" className="hover:underline flex items-center">
                  <BookOpen className="mr-1 h-4 w-4" />
                  About
                </Link>
                <Link to="/methodology" className="hover:underline flex items-center">
                  <FileText className="mr-1 h-4 w-4" />
                  Methodology
                </Link>
                <Link to="/data" className="hover:underline flex items-center">
                  <Database className="mr-1 h-4 w-4" />
                  Data
                </Link>
                <Link to="/models" className="hover:underline flex items-center">
                  <BarChart2 className="mr-1 h-4 w-4" />
                  Models
                </Link>
                <Link to="/recommendations" className="hover:underline flex items-center">
                  <AlertTriangle className="mr-1 h-4 w-4" />
                  Recommendations
                </Link>
              </nav>
              
              {/* Mobile Menu Button */}
              <button 
                className="md:hidden text-primary-foreground"
                onClick={toggleMobileMenu}
                aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
              >
                {mobileMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
              </button>
            </div>
          </header>
          
          {/* Mobile Navigation */}
          {mobileMenuOpen && (
            <div className="md:hidden bg-background border-b">
              <nav className="container mx-auto py-4 px-6 flex flex-col space-y-4">
                <Link to="/" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <Home className="mr-2 h-5 w-5" />
                  Home
                </Link>
                <Link to="/about" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <BookOpen className="mr-2 h-5 w-5" />
                  About
                </Link>
                <Link to="/methodology" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <FileText className="mr-2 h-5 w-5" />
                  Methodology
                </Link>
                <Link to="/environmental-impact" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <AlertTriangle className="mr-2 h-5 w-5" />
                  Environmental Impact
                </Link>
                <Link to="/health-impact" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <AlertTriangle className="mr-2 h-5 w-5" />
                  Health Impact
                </Link>
                <Link to="/socioeconomic-impact" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <Users className="mr-2 h-5 w-5" />
                  Socioeconomic Impact
                </Link>
                <Link to="/data" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <Database className="mr-2 h-5 w-5" />
                  Data
                </Link>
                <Link to="/models" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <BarChart2 className="mr-2 h-5 w-5" />
                  Models
                </Link>
                <Link to="/recommendations" className="hover:underline flex items-center" onClick={closeMobileMenu}>
                  <AlertTriangle className="mr-2 h-5 w-5" />
                  Recommendations
                </Link>
              </nav>
            </div>
          )}
          
          {/* Main Content */}
          <main className="flex-grow">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/methodology" element={<MethodologyPage />} />
              <Route path="/environmental-impact" element={<EnvironmentalImpactPage />} />
              <Route path="/health-impact" element={<HealthImpactPage />} />
              <Route path="/socioeconomic-impact" element={<SocioeconomicImpactPage />} />
              <Route path="/recommendations" element={<RecommendationsPage />} />
              <Route path="/models" element={<ModelsPage />} />
              <Route path="/data" element={<DataPage />} />
            </Routes>
          </main>
          
          {/* Footer */}
          <footer className="bg-muted py-8 px-6">
            <div className="container mx-auto">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div>
                  <h3 className="text-lg font-semibold mb-4">Jadugora Impact Assessment</h3>
                  <p className="text-sm text-muted-foreground">
                    A comprehensive data science project analyzing the environmental, health, and societal 
                    consequences of uranium mining in Jadugora, Jharkhand, India.
                  </p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
                  <ul className="space-y-2 text-sm">
                    <li><Link to="/" className="hover:underline">Home</Link></li>
                    <li><Link to="/about" className="hover:underline">About</Link></li>
                    <li><Link to="/methodology" className="hover:underline">Methodology</Link></li>
                    <li><Link to="/data" className="hover:underline">Data</Link></li>
                    <li><Link to="/models" className="hover:underline">Models</Link></li>
                    <li><Link to="/recommendations" className="hover:underline">Recommendations</Link></li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-4">Impact Areas</h3>
                  <ul className="space-y-2 text-sm">
                    <li><Link to="/environmental-impact" className="hover:underline">Environmental Impact</Link></li>
                    <li><Link to="/health-impact" className="hover:underline">Health Impact</Link></li>
                    <li><Link to="/socioeconomic-impact" className="hover:underline">Socioeconomic Impact</Link></li>
                  </ul>
                </div>
              </div>
              <div className="mt-8 pt-8 border-t border-border text-center text-sm text-muted-foreground">
                <p>© {new Date().getFullYear()} Jadugora Impact Assessment Project. All rights reserved.</p>
                <p className="mt-2">
                  This website presents findings from a data science project on uranium mining impacts. 
                  For official information, please consult relevant government and regulatory authorities.
                </p>
              </div>
            </div>
          </footer>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
