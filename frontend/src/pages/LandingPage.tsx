
import { motion } from 'framer-motion';
import { Brain, Activity, Target, Eye, Battery, ChevronRight } from 'lucide-react';

const LandingPage = ({ onGetStarted, onLogin }: { onGetStarted: () => void; onLogin: () => void }) => {
  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans selection:bg-sage selection:text-white overflow-x-hidden">
      {/* Glassmorphism Navigation */}
      <nav className="fixed w-full z-50 top-0 transition-all duration-300 bg-ivory/70 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-8 h-8 text-plum" />
            <span className="text-xl font-bold tracking-tight text-gray-900">MHBAP</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-gray-600">
            <a href="#features" className="hover:text-plum transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-plum transition-colors">How it Works</a>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={onLogin} className="text-sm font-medium text-gray-600 hover:text-plum transition-colors">Log In</button>
            <button 
              onClick={onGetStarted}
              className="bg-plum text-white px-6 py-2.5 rounded-full text-sm font-medium hover:bg-plum-dark transition-all hover:shadow-lg hover:shadow-plum/30 flex items-center gap-2"
            >
              Get Started <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[800px] bg-gradient-to-b from-plum/5 to-transparent pointer-events-none" />
        <div className="max-w-7xl mx-auto px-6 relative">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-8 items-center">
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="max-w-2xl"
            >
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-plum/10 text-plum font-semibold text-sm mb-6">
                <Brain className="w-4 h-4" />
                <span>Research-Grade Platform</span>
              </div>
              <h1 className="text-5xl lg:text-7xl font-extrabold tracking-tight text-gray-900 leading-[1.1] mb-6">
                Analyze Human Behaviour <br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-plum to-sage">Multimodally</span>
              </h1>
              <p className="text-lg text-gray-600 mb-10 leading-relaxed max-w-xl">
                A unified environment for integrating facial expressions, voice, and physiological data to evaluate emotion, engagement, and cognitive states.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 items-center">
                <button 
                  onClick={onGetStarted}
                  className="w-full sm:w-auto bg-plum text-white px-8 py-3.5 rounded-full text-base font-semibold hover:bg-plum-dark transition-all hover:shadow-xl hover:shadow-plum/30 flex items-center justify-center gap-2"
                >
                  Launch Dashboard <ChevronRight className="w-4 h-4" />
                </button>
                <a href="#features" className="w-full sm:w-auto text-gray-600 font-semibold px-6 py-3.5 hover:text-plum transition-colors flex items-center justify-center">
                  Explore Features
                </a>
              </div>
            </motion.div>
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
              className="relative lg:ml-auto"
            >
              <div className="absolute inset-0 bg-gradient-to-tr from-plum/20 to-sage/20 rounded-3xl blur-3xl" />
              <img 
                src="/multimodal-hero.png" 
                alt="Multimodal analysis visualization" 
                className="relative z-10 w-full max-w-lg rounded-3xl shadow-2xl border border-white/20 object-cover"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 bg-white relative">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold mb-4 text-gray-900">Comprehensive Analysis</h2>
            <p className="text-gray-600 text-lg">Our multimodal approach captures nuance that single-modality systems miss.</p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              { icon: Activity, title: 'Emotion & Stress', desc: 'Real-time affective computing combining facial expression and physiological signals.', color: 'text-plum', bg: 'bg-plum/10' },
              { icon: Target, title: 'Engagement Tracking', desc: 'Measure user involvement through behavioral cues and interaction patterns.', color: 'text-sage-dark', bg: 'bg-sage/20' },
              { icon: Eye, title: 'Visual Attention', desc: 'Precise gaze tracking and fixation analysis to understand visual focus.', color: 'text-blue-600', bg: 'bg-blue-100' },
              { icon: Battery, title: 'Cognitive Fatigue', desc: 'Detect early signs of mental exhaustion before it impacts performance.', color: 'text-amber-600', bg: 'bg-amber-100' }
            ].map((feature, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                className="p-8 rounded-2xl bg-ivory border border-gray-100 hover:shadow-lg hover:-translate-y-1 transition-all"
              >
                <div className={`w-14 h-14 rounded-xl flex items-center justify-center mb-6 ${feature.bg} ${feature.color}`}>
                  <feature.icon className="w-7 h-7" />
                </div>
                <h3 className="text-xl font-bold mb-3 text-gray-900">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24 bg-gray-900 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-plum/20 rounded-full blur-[120px] pointer-events-none opacity-50" />
        <div className="max-w-7xl mx-auto px-6 relative z-10">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl lg:text-4xl font-bold mb-6">Built for Researchers and Innovators</h2>
              <p className="text-gray-400 text-lg mb-8 leading-relaxed">
                Connect multiple data streams effortlessly. Our engine synchronizes video, audio, and physiological data with microsecond precision, delivering clean, actionable metrics.
              </p>
              <ul className="space-y-6">
                {['Connect data sources securely', 'Real-time synchronization engine', 'Extract multimodal features', 'Generate comprehensive insights'].map((step, i) => (
                  <motion.li 
                    key={i}
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                    className="flex items-center gap-4"
                  >
                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-plum flex items-center justify-center font-bold text-sm">
                      {i + 1}
                    </div>
                    <span className="text-gray-200 font-medium">{step}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              className="bg-gray-800 rounded-2xl border border-gray-700 p-8 shadow-2xl relative"
            >
               {/* Abstract representation of a dashboard or code */}
               <div className="flex gap-2 mb-6 border-b border-gray-700 pb-4">
                 <div className="w-3 h-3 rounded-full bg-red-500" />
                 <div className="w-3 h-3 rounded-full bg-yellow-500" />
                 <div className="w-3 h-3 rounded-full bg-green-500" />
               </div>
               <div className="space-y-4 font-mono text-sm">
                  <div className="text-sage">import {'{ MultimodalAnalyzer }'} from '@mhbap/core';</div>
                  <div className="text-gray-400">const analyzer = new MultimodalAnalyzer({'{'}</div>
                  <div className="text-plum-light ml-4">modalities: ['video', 'audio', 'eeg'],</div>
                  <div className="text-plum-light ml-4">precision: 'high',</div>
                  <div className="text-gray-400">{'}'});</div>
                  <br/>
                  <div className="text-gray-400">analyzer.on('insight', (data) =&gt; {'{'}</div>
                  <div className="text-blue-400 ml-4">console.log(data.engagementScore);</div>
                  <div className="text-gray-400">{'}'});</div>
               </div>
            </motion.div>
          </div>
        </div>
      </section>


      {/* Footer */}
      <footer className="bg-gray-950 text-gray-400 py-12 border-t border-gray-800 mt-auto">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center md:items-end gap-6 text-center md:text-left">
          <div className="flex flex-col items-center md:items-start gap-1">
            <div className="flex items-center gap-2 mb-2">
              <Brain className="w-6 h-6 text-plum" />
              <span className="text-xl font-bold tracking-tight text-white">MHBAP</span>
            </div>
            <p className="text-gray-300 font-medium">Multimodal Human Behaviour Analysis Platform</p>
            <p className="text-sm text-gray-500">Research Platform • UET Lahore</p>
            <p className="text-sm text-gray-600 mt-2">Developed by Ashhal Kaleem</p>
          </div>
          
          <div className="text-sm text-gray-500 md:text-right mt-4 md:mt-0">
            <p>© {new Date().getFullYear()} MHBAP. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
