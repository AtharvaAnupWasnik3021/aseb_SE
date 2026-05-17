import { useEffect, useRef, useState, useCallback } from 'react'
import {
  LineChart, Line, AreaChart, Area, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PolarRadiusAxis, Cell, BarChart, Bar
} from 'recharts'
import {
  Dna, Brain, Microscope, Zap, Target, Search, FlaskConical, ChevronRight,
  ArrowRight, Activity, Shield, Star, Users, TrendingUp, Clock, DollarSign,
  CheckCircle2, XCircle, Circle, Cpu, Network, Globe, Lock, BarChart2,
  Layers, GitBranch, Atom, Beaker, Sigma, Bot, Sparkles, Eye, Play,
  ChevronDown, Menu, X, ExternalLink, ArrowLeft
} from 'lucide-react'

// ===== CUSTOM CURSOR =====
const CustomCursor = () => {
  const cursorRef = useRef(null)
  const followerRef = useRef(null)

  useEffect(() => {
    const moveCursor = (e) => {
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate(${e.clientX - 6}px, ${e.clientY - 6}px)`
      }
      if (followerRef.current) {
        followerRef.current.style.transform = `translate(${e.clientX - 20}px, ${e.clientY - 20}px)`
      }
    }
    window.addEventListener('mousemove', moveCursor)
    return () => window.removeEventListener('mousemove', moveCursor)
  }, [])

  return (
    <>
      <div ref={cursorRef} className="cursor" />
      <div ref={followerRef} className="cursor-follower" />
    </>
  )
}

// ===== SCROLL PROGRESS =====
const ScrollProgress = () => {
  const [progress, setProgress] = useState(0)
  useEffect(() => {
    const handleScroll = () => {
      const total = document.documentElement.scrollHeight - window.innerHeight
      setProgress((window.scrollY / total) * 100)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])
  return <div className="scroll-progress" style={{ width: `${progress}%` }} />
}

// ===== ANIMATED PARTICLES =====
const Particles = ({ count = 30 }) => {
  const particles = Array.from({ length: count }, (_, i) => ({
    id: i,
    left: `${Math.random() * 100}%`,
    delay: `${Math.random() * 8}s`,
    duration: `${6 + Math.random() * 8}s`,
    size: `${1 + Math.random() * 3}px`,
    color: ['#00f5ff', '#0080ff', '#8b5cf6', '#10b981'][Math.floor(Math.random() * 4)],
  }))

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((p) => (
        <div
          key={p.id}
          className="absolute rounded-full"
          style={{
            left: p.left,
            bottom: '-10px',
            width: p.size,
            height: p.size,
            background: p.color,
            boxShadow: `0 0 6px ${p.color}`,
            animation: `particle-float ${p.duration} ${p.delay} linear infinite`,
          }}
        />
      ))}
    </div>
  )
}

// ===== BLOB BACKGROUND =====
const BlobBackground = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    <div className="blob" style={{ width: '600px', height: '600px', top: '-100px', left: '-100px', background: 'rgba(0,128,255,0.06)', animationDelay: '0s' }} />
    <div className="blob" style={{ width: '500px', height: '500px', top: '30%', right: '-80px', background: 'rgba(139,92,246,0.06)', animationDelay: '5s' }} />
    <div className="blob" style={{ width: '400px', height: '400px', bottom: '10%', left: '20%', background: 'rgba(0,245,255,0.05)', animationDelay: '10s' }} />
  </div>
)

// ===== SECTION REVEAL HOOK =====
const useReveal = () => {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible')
          }
        })
      },
      { threshold: 0.1 }
    )
    document.querySelectorAll('.reveal, .reveal-left, .reveal-right').forEach((el) => observer.observe(el))
    return () => observer.disconnect()
  }, [])
}

// ===== COUNTER HOOK =====
const useCounter = (target, duration = 2000, suffix = '') => {
  const [count, setCount] = useState(0)
  const [started, setStarted] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting && !started) {
        setStarted(true)
        let start = 0
        const step = target / (duration / 16)
        const timer = setInterval(() => {
          start += step
          if (start >= target) { setCount(target); clearInterval(timer) }
          else setCount(Math.floor(start))
        }, 16)
      }
    }, { threshold: 0.5 })
    if (ref.current) observer.observe(ref.current)
    return () => observer.disconnect()
  }, [target, duration, started])

  return { count, ref }
}

// ===== NAVBAR =====
const Navbar = ({ onOpenConsole }) => {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const links = ['Platform', 'Research', 'Technology', 'Pipeline', 'Impact']

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${scrolled ? 'py-3' : 'py-5'}`}>
      <div className={`mx-4 rounded-xl transition-all duration-500 ${scrolled ? 'glass-card border-glow-cyan' : ''}`}>
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative w-9 h-9">
              <div className="absolute inset-0 rounded-lg border border-cyan-400/40 bg-cyan-400/10 flex items-center justify-center">
                <Dna size={18} className="text-cyan-400" />
              </div>
              <div className="absolute inset-0 rounded-lg border border-cyan-400/20 animate-ping" style={{ animationDuration: '3s' }} />
            </div>
            <div>
              <span className="font-display text-white font-bold text-base tracking-wider">PROTEIN</span>
              <span className="font-display font-bold text-base tracking-wider" style={{ color: '#00f5ff' }}>AI</span>
            </div>
          </div>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-8">
            {links.map((link) => (
              <a key={link} href={`#${link.toLowerCase()}`}
                className="font-body text-sm font-500 tracking-widest text-gray-400 hover:text-cyan-400 transition-colors duration-300 uppercase">
                {link}
              </a>
            ))}
          </div>

          {/* CTA */}
          <div className="hidden md:flex items-center gap-3">
            <button className="btn-secondary text-xs" onClick={onOpenConsole}>Open Console</button>
            <button className="btn-primary text-xs" onClick={onOpenConsole}>Launch Platform</button>
          </div>

          {/* Mobile Menu Toggle */}
          <button className="md:hidden text-gray-400 hover:text-white" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {menuOpen && (
          <div className="md:hidden glass-card m-2 p-4 rounded-xl">
            {links.map((link) => (
              <a key={link} href={`#${link.toLowerCase()}`}
                className="block py-2 font-body text-sm font-500 tracking-widest text-gray-400 hover:text-cyan-400 uppercase"
                onClick={() => setMenuOpen(false)}>
                {link}
              </a>
            ))}
            <div className="flex gap-3 mt-4">
              <button className="btn-secondary text-xs flex-1" onClick={onOpenConsole}>Console</button>
              <button className="btn-primary text-xs flex-1" onClick={onOpenConsole}>Launch</button>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}

// ===== PROTEIN MOLECULE SVG =====
const ProteinMolecule = () => (
  <div className="relative w-full h-full flex items-center justify-center">
    {/* Central glow */}
    <div className="absolute w-64 h-64 rounded-full"
      style={{ background: 'radial-gradient(circle, rgba(0,245,255,0.12) 0%, transparent 70%)' }} />

    {/* Rotating rings */}
    <div className="absolute w-80 h-80 rounded-full border border-cyan-400/10"
      style={{ animation: 'ring-rotate 20s linear infinite' }} />
    <div className="absolute w-56 h-56 rounded-full border border-violet-500/10"
      style={{ animation: 'ring-rotate-reverse 15s linear infinite' }} />
    <div className="absolute w-40 h-40 rounded-full border border-blue-500/15"
      style={{ animation: 'ring-rotate 10s linear infinite' }} />

    {/* SVG molecular structure */}
    <svg width="320" height="320" viewBox="0 0 320 320" className="relative z-10">
      <defs>
        <radialGradient id="nodeGrad1" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#00f5ff" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#0080ff" stopOpacity="0.3" />
        </radialGradient>
        <radialGradient id="nodeGrad2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.2" />
        </radialGradient>
        <radialGradient id="nodeGrad3" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.2" />
        </radialGradient>
        <filter id="glow1">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Bonds */}
      {[
        [160,160,80,80],[160,160,240,80],[160,160,60,180],[160,160,260,180],
        [160,160,100,260],[160,160,220,260],[80,80,60,180],[240,80,260,180],
        [60,180,100,260],[260,180,220,260],[100,260,220,260],[80,80,240,80]
      ].map(([x1,y1,x2,y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="rgba(0,245,255,0.2)" strokeWidth="1"
          strokeDasharray="4 3"
          style={{ animation: `flow 2s linear ${i * 0.2}s infinite` }}
        />
      ))}

      {/* Secondary bonds */}
      {[[80,80,100,260],[240,80,100,260],[60,180,220,260]].map(([x1,y1,x2,y2], i) => (
        <line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="rgba(139,92,246,0.15)" strokeWidth="1" strokeDasharray="2 6"
        />
      ))}

      {/* Central node */}
      <circle cx="160" cy="160" r="20" fill="url(#nodeGrad1)" filter="url(#glow1)"
        style={{ animation: 'neural-pulse 2s ease-in-out infinite' }} />
      <circle cx="160" cy="160" r="28" fill="none" stroke="rgba(0,245,255,0.3)" strokeWidth="1"
        style={{ animation: 'ring-rotate 4s linear infinite' }} />

      {/* Outer nodes */}
      {[
        [80,80,'url(#nodeGrad2)',12],[240,80,'url(#nodeGrad3)',10],
        [60,180,'url(#nodeGrad1)',14],[260,180,'url(#nodeGrad2)',10],
        [100,260,'url(#nodeGrad3)',12],[220,260,'url(#nodeGrad1)',11],
      ].map(([cx,cy,fill,r], i) => (
        <g key={i}>
          <circle cx={cx} cy={cy} r={r} fill={fill} filter="url(#glow1)"
            style={{ animation: `neural-pulse ${2 + i * 0.3}s ease-in-out ${i * 0.4}s infinite` }} />
          <circle cx={cx} cy={cy} r={r + 6} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
        </g>
      ))}

      {/* Floating labels */}
      {[
        [80,65,'HIS'],[240,65,'ASP'],[38,185,'GLY'],[272,185,'PHE'],
        [85,275,'TYR'],[225,275,'SER']
      ].map(([x,y,label], i) => (
        <text key={i} x={x} y={y} fill="rgba(0,245,255,0.6)" fontSize="9"
          fontFamily="JetBrains Mono" textAnchor="middle">{label}</text>
      ))}

      {/* Center label */}
      <text x="160" y="155" fill="rgba(0,245,255,0.9)" fontSize="8"
        fontFamily="JetBrains Mono" textAnchor="middle">ACTIVE</text>
      <text x="160" y="166" fill="rgba(0,245,255,0.9)" fontSize="8"
        fontFamily="JetBrains Mono" textAnchor="middle">SITE</text>
    </svg>

    {/* Orbiting particles */}
    <div className="absolute w-80 h-80">
      {[0,1,2].map((i) => (
        <div key={i} className="absolute top-1/2 left-1/2"
          style={{
            width: '8px', height: '8px', marginTop: '-4px', marginLeft: '-4px',
            animation: `orbit ${6 + i * 3}s linear ${i * 2}s infinite`,
          }}>
          <div className="w-2 h-2 rounded-full"
            style={{ background: ['#00f5ff','#8b5cf6','#10b981'][i],
              boxShadow: `0 0 8px ${['#00f5ff','#8b5cf6','#10b981'][i]}` }} />
        </div>
      ))}
    </div>
  </div>
)

// ===== HERO SECTION =====
const Hero = ({ onOpenConsole }) => {
  const stats = [
    { label: 'Proteins Analyzed', value: '2.4M+', color: '#00f5ff' },
    { label: 'Accuracy Rate', value: '94.7%', color: '#10b981' },
    { label: 'Discovery Speed', value: '100x', color: '#8b5cf6' },
  ]

  const badges = [
    { text: 'AlphaFold Integration', icon: <Dna size={12} /> },
    { text: 'Multi-Agent Reasoning', icon: <Brain size={12} /> },
    { text: 'Real-Time Analysis', icon: <Zap size={12} /> },
  ]

  return (
    <section id="platform" className="relative min-h-screen flex items-center overflow-hidden pt-24">
      {/* Animated grid */}
      <div className="absolute inset-0 grid-bg opacity-50" />
      <BlobBackground />
      <Particles count={25} />

      {/* Radial glow center */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] rounded-full"
        style={{ background: 'radial-gradient(circle, rgba(0,128,255,0.06) 0%, transparent 70%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6 w-full">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left: Text */}
          <div>
            {/* Status badge */}
            <div className="inline-flex items-center gap-2 glass-card border-glow-cyan px-4 py-2 rounded-full mb-8">
              <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
              <span className="section-label">Live Platform • Version 3.2.1</span>
            </div>

            {/* Main headline */}
            <h1 className="font-display font-black leading-tight mb-6">
              <span className="block text-5xl md:text-6xl lg:text-7xl text-white mb-2">
                Reinventing
              </span>
              <span className="block text-5xl md:text-6xl lg:text-7xl gradient-text-primary mb-2">
                Drug Discovery
              </span>
              <span className="block text-5xl md:text-6xl lg:text-7xl text-white">
                with <span className="gradient-text-warm">Agentic AI</span>
              </span>
            </h1>

            <p className="font-body text-lg text-gray-400 leading-relaxed mb-8 max-w-lg">
              Autonomous intelligence for protein drugability prediction, molecular reasoning, and accelerated therapeutic discovery. Compress decades of research into days.
            </p>

            {/* Buttons */}
            <div className="flex flex-wrap gap-4 mb-12">
              <button className="btn-primary flex items-center gap-2" onClick={onOpenConsole}>
                <Play size={14} />
                Launch Platform
              </button>
              <a href="#research" className="btn-secondary flex items-center gap-2">
                <ExternalLink size={14} />
                View Research
              </a>
            </div>

            {/* Floating badges */}
            <div className="flex flex-wrap gap-3 mb-10">
              {badges.map((badge, i) => (
                <div key={i} className="flex items-center gap-2 glass-card px-3 py-2 rounded-full border border-cyan-400/10 hover:border-cyan-400/30 transition-colors">
                  <span className="text-cyan-400">{badge.icon}</span>
                  <span className="font-mono text-xs text-gray-300">{badge.text}</span>
                </div>
              ))}
            </div>

            {/* Live stats */}
            <div className="grid grid-cols-3 gap-4">
              {stats.map((s, i) => (
                <div key={i} className="glass-card p-4 rounded-xl border-glow-cyan relative overflow-hidden">
                  <div className="shimmer absolute inset-0 rounded-xl" />
                  <div className="relative z-10">
                    <div className="font-display font-black text-2xl" style={{ color: s.color }}>{s.value}</div>
                    <div className="section-label text-gray-500 text-xs mt-1">{s.label}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Protein visualization */}
          <div className="relative hidden lg:block">
            <div className="relative w-full h-[600px]">
              {/* Outer decorative rings */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="absolute w-[500px] h-[500px] rounded-full border border-cyan-400/5"
                  style={{ animation: 'ring-rotate 30s linear infinite' }} />
                <div className="absolute w-[420px] h-[420px] rounded-full border border-violet-500/5"
                  style={{ animation: 'ring-rotate-reverse 25s linear infinite' }} />
              </div>

              {/* Scan line overlay */}
              <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
                <div className="scan-line" />
              </div>

              <ProteinMolecule />

              {/* Floating info cards */}
              <div className="absolute top-16 right-0 glass-card border-glow-cyan p-3 rounded-xl w-48">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
                  <span className="font-mono text-xs text-emerald-400">ANALYZING</span>
                </div>
                <div className="font-display text-xs text-white">BCL-2 Protein</div>
                <div className="w-full h-1 bg-gray-800 rounded mt-2">
                  <div className="h-full bg-gradient-to-r from-cyan-400 to-blue-500 rounded loading-bar" style={{ width: '78%' }} />
                </div>
                <div className="font-mono text-xs text-gray-400 mt-1">Drugability: 94.2%</div>
              </div>

              <div className="absolute bottom-24 left-0 glass-card border-glow-violet p-3 rounded-xl w-44">
                <div className="section-label text-xs mb-1">BINDING POCKETS</div>
                <div className="font-display text-2xl font-black" style={{ color: '#8b5cf6' }}>7</div>
                <div className="flex gap-1 mt-2">
                  {[0.9,0.7,0.8,0.6,0.5,0.85,0.95].map((v, i) => (
                    <div key={i} className="flex-1 rounded"
                      style={{ height: `${v * 24}px`, background: `rgba(139,92,246,${v * 0.8})`, alignSelf: 'flex-end' }} />
                  ))}
                </div>
              </div>

              <div className="absolute top-1/2 -right-4 glass-card p-3 rounded-xl w-36">
                <div className="section-label text-xs mb-2">AI CONFIDENCE</div>
                <div className="space-y-1">
                  {['Structure','Binding','Toxicity'].map((label, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-xs text-gray-400 mb-0.5">
                        <span>{label}</span>
                        <span className="text-cyan-400">{[94,87,91][i]}%</span>
                      </div>
                      <div className="w-full h-1 bg-gray-800 rounded">
                        <div className="h-full rounded"
                          style={{ width: `${[94,87,91][i]}%`, background: `rgba(0,245,255,${0.6 + i*0.1})` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-60">
          <span className="section-label text-xs">SCROLL TO EXPLORE</span>
          <ChevronDown size={16} className="text-cyan-400 animate-bounce" />
        </div>
      </div>
    </section>
  )
}

// ===== PROBLEM SECTION =====
const Problem = () => {
  useReveal()

  const problems = [
    { icon: <DollarSign size={28} />, value: '$2.6B', label: 'Average Development Cost', desc: 'Per approved drug, including failed trials', color: '#f0abfc', delay: '0s' },
    { icon: <XCircle size={28} />, value: '90%', label: 'Clinical Trial Failure', desc: 'Compounds that never reach patients', color: '#ff6b6b', delay: '0.1s' },
    { icon: <Clock size={28} />, value: '12 Years', label: 'Development Timeline', desc: 'From target to market approval', color: '#fbbf24', delay: '0.2s' },
    { icon: <Search size={28} />, value: '20,000+', label: 'Druggable Proteins', desc: 'Vast unexplored search space', color: '#00f5ff', delay: '0.3s' },
  ]

  const timelineData = [
    { year: '2005', cost: 12, time: 9 },
    { year: '2010', cost: 18, time: 10 },
    { year: '2015', cost: 21, time: 11 },
    { year: '2020', cost: 24, time: 12 },
    { year: '2025', cost: 26, time: 12.5 },
    { year: '2026*', cost: 8, time: 3 },
  ]

  return (
    <section id="problem" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <BlobBackground />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// THE CHALLENGE</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Traditional Drug Discovery is
            <span className="gradient-text-warm"> Broken</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            The pharmaceutical industry faces a systemic crisis. Decades of research, billions in capital, and yet failure rates continue to climb.
          </p>
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-16">
          {problems.map((p, i) => (
            <div key={i} className="glass-card feature-card p-6 rounded-2xl reveal gradient-border"
              style={{ transitionDelay: p.delay }}>
              <div className="mb-4 p-3 rounded-xl inline-flex" style={{ background: `${p.color}15` }}>
                <span style={{ color: p.color }}>{p.icon}</span>
              </div>
              <div className="font-display font-black text-3xl mb-1" style={{ color: p.color }}>{p.value}</div>
              <div className="font-body font-600 text-white text-sm mb-1">{p.label}</div>
              <div className="font-body text-xs text-gray-500">{p.desc}</div>
            </div>
          ))}
        </div>

        {/* Chart section */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Rising cost chart */}
          <div className="glass-card p-6 rounded-2xl reveal-left">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="section-label mb-1">COST TRAJECTORY</div>
                <div className="font-display font-bold text-white text-lg">R&D Cost Per Drug ($B)</div>
              </div>
              <div className="flex items-center gap-2 text-red-400">
                <TrendingUp size={16} />
                <span className="font-mono text-xs">+117% since 2005</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={timelineData}>
                <defs>
                  <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f0abfc" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#f0abfc" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="year" stroke="rgba(255,255,255,0.2)" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.5)' }} />
                <YAxis stroke="rgba(255,255,255,0.2)" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.5)' }} />
                <Tooltip contentStyle={{ background: '#0d1530', border: '1px solid rgba(0,245,255,0.2)', borderRadius: '8px', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Area type="monotone" dataKey="cost" stroke="#f0abfc" fill="url(#costGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
            <div className="mt-3 flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-emerald-400" />
              <span className="font-mono text-xs text-emerald-400">2026* = ProteinAI projection</span>
            </div>
          </div>

          {/* Timeline visualization */}
          <div className="glass-card p-6 rounded-2xl reveal-right">
            <div className="section-label mb-1">DEVELOPMENT PIPELINE</div>
            <div className="font-display font-bold text-white text-lg mb-5">Traditional vs AI-Accelerated</div>
            <div className="space-y-4">
              {[
                { phase: 'Target ID', trad: 3, ai: 0.1, color: '#00f5ff' },
                { phase: 'Lead Discovery', trad: 2, ai: 0.3, color: '#0080ff' },
                { phase: 'Optimization', trad: 3, ai: 0.5, color: '#8b5cf6' },
                { phase: 'Pre-Clinical', trad: 2, ai: 0.8, color: '#10b981' },
              ].map((item, i) => (
                <div key={i}>
                  <div className="flex justify-between mb-1">
                    <span className="font-body text-sm text-gray-300">{item.phase}</span>
                    <span className="font-mono text-xs" style={{ color: item.color }}>
                      {item.ai}yr vs {item.trad}yr
                    </span>
                  </div>
                  <div className="relative h-3 bg-gray-800 rounded-full overflow-hidden">
                    <div className="absolute left-0 top-0 h-full rounded-full opacity-30"
                      style={{ width: `${(item.trad / 3) * 100}%`, background: '#666' }} />
                    <div className="absolute left-0 top-0 h-full rounded-full"
                      style={{ width: `${(item.ai / 3) * 100}%`, background: item.color,
                        boxShadow: `0 0 8px ${item.color}` }} />
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-5 p-3 rounded-xl bg-emerald-400/5 border border-emerald-400/20">
              <div className="flex items-center gap-2">
                <Zap size={14} className="text-emerald-400" />
                <span className="font-mono text-xs text-emerald-400">ProteinAI reduces total timeline by 10-15x</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ===== AGENTIC AI SECTION =====
const AgenticAI = () => {
  useReveal()

  const comparisons = [
    { feature: 'Autonomous reasoning', trad: false, agent: true },
    { feature: 'Multi-step inference', trad: false, agent: true },
    { feature: 'Adaptive analysis', trad: false, agent: true },
    { feature: 'Real-time decisions', trad: false, agent: true },
    { feature: 'Biological context', trad: 'Limited', agent: true },
    { feature: 'Self-correcting loops', trad: false, agent: true },
    { feature: 'Cross-domain reasoning', trad: false, agent: true },
    { feature: 'Explainable outputs', trad: 'Partial', agent: true },
  ]

  const agents = [
    { name: 'Structure Agent', icon: <Atom size={20} />, role: 'Parses 3D protein topology', color: '#00f5ff' },
    { name: 'Pocket Agent', icon: <Search size={20} />, role: 'Detects binding cavities', color: '#8b5cf6' },
    { name: 'Chemistry Agent', icon: <FlaskConical size={20} />, role: 'Scores molecular interactions', color: '#10b981' },
    { name: 'Pathway Agent', icon: <Network size={20} />, role: 'Maps biological cascades', color: '#f0abfc' },
    { name: 'Drug Agent', icon: <Target size={20} />, role: 'Predicts drug viability', color: '#fbbf24' },
    { name: 'Safety Agent', icon: <Shield size={20} />, role: 'Assesses toxicity risks', color: '#0080ff' },
  ]

  return (
    <section id="technology" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-20" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 20% 50%, rgba(139,92,246,0.06) 0%, transparent 60%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// AGENTIC INTELLIGENCE</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Beyond AI. <span className="gradient-text-primary">Autonomous Reasoning.</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            Agentic AI doesn't just predict — it reasons, adapts, and self-corrects across multiple biological domains simultaneously.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12 mb-16">
          {/* Comparison table */}
          <div className="glass-card p-6 rounded-2xl reveal-left">
            <div className="section-label mb-4">CAPABILITY COMPARISON</div>
            <div className="grid grid-cols-3 gap-2 mb-3 pb-3 border-b border-white/5">
              <div className="font-body font-600 text-gray-400 text-sm">Capability</div>
              <div className="text-center font-mono text-xs text-red-400">Traditional ML</div>
              <div className="text-center font-mono text-xs text-cyan-400">ProteinAI Agent</div>
            </div>
            <div className="space-y-3">
              {comparisons.map((c, i) => (
                <div key={i} className="grid grid-cols-3 gap-2 items-center py-1">
                  <span className="font-body text-sm text-gray-300">{c.feature}</span>
                  <div className="flex justify-center">
                    {c.trad === false ? (
                      <XCircle size={16} className="cross-dim" />
                    ) : c.trad === true ? (
                      <CheckCircle2 size={16} className="check-glow" />
                    ) : (
                      <span className="font-mono text-xs text-yellow-500">{c.trad}</span>
                    )}
                  </div>
                  <div className="flex justify-center">
                    <CheckCircle2 size={16} className="check-glow" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Agent network visualization */}
          <div className="glass-card p-6 rounded-2xl reveal-right">
            <div className="section-label mb-4">MULTI-AGENT ORCHESTRATION</div>
            <div className="relative">
              {/* Central hub */}
              <div className="flex justify-center mb-8">
                <div className="relative">
                  <div className="w-20 h-20 rounded-full border-2 border-cyan-400/40 bg-cyan-400/5 flex items-center justify-center">
                    <Brain size={32} className="text-cyan-400" />
                  </div>
                  <div className="absolute inset-0 rounded-full border border-cyan-400/20"
                    style={{ animation: 'ring-rotate 4s linear infinite' }} />
                  <div className="absolute -inset-2 rounded-full border border-cyan-400/10"
                    style={{ animation: 'ring-rotate-reverse 6s linear infinite' }} />
                </div>
              </div>
              {/* Agent cards */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {agents.map((agent, i) => (
                  <div key={i} className="p-3 rounded-xl border transition-all duration-300 hover:scale-105"
                    style={{
                      background: `${agent.color}08`,
                      borderColor: `${agent.color}25`,
                    }}>
                    <div className="flex items-center gap-2 mb-1">
                      <span style={{ color: agent.color }}>{agent.icon}</span>
                      <span className="font-display text-xs font-bold text-white">{agent.name}</span>
                    </div>
                    <span className="font-mono text-xs text-gray-500">{agent.role}</span>
                    <div className="mt-2 flex items-center gap-1">
                      <div className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: agent.color }} />
                      <span className="font-mono text-xs" style={{ color: agent.color }}>ACTIVE</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Radar chart */}
        <div className="glass-card p-8 rounded-2xl reveal">
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            <div>
              <div className="section-label mb-3">PERFORMANCE RADAR</div>
              <h3 className="font-display font-black text-2xl text-white mb-4">
                Comprehensive Capability Assessment
              </h3>
              <div className="space-y-3">
                {[
                  { label: 'Prediction Accuracy', value: 94, color: '#00f5ff' },
                  { label: 'Processing Speed', value: 98, color: '#10b981' },
                  { label: 'Biological Coverage', value: 89, color: '#8b5cf6' },
                  { label: 'Explainability', value: 92, color: '#f0abfc' },
                ].map((m, i) => (
                  <div key={i}>
                    <div className="flex justify-between mb-1">
                      <span className="font-body text-sm text-gray-300">{m.label}</span>
                      <span className="font-mono text-xs" style={{ color: m.color }}>{m.value}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-800 rounded-full">
                      <div className="h-full rounded-full" style={{ width: `${m.value}%`, background: m.color, boxShadow: `0 0 6px ${m.color}` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <ResponsiveContainer width="100%" height={280}>
                <RadarChart data={[
                  { metric: 'Structure', AI: 94, Traditional: 45 },
                  { metric: 'Binding', AI: 91, Traditional: 38 },
                  { metric: 'Toxicity', AI: 88, Traditional: 52 },
                  { metric: 'Pathway', AI: 96, Traditional: 30 },
                  { metric: 'Speed', AI: 98, Traditional: 15 },
                  { metric: 'Coverage', AI: 89, Traditional: 42 },
                ]}>
                  <PolarGrid stroke="rgba(255,255,255,0.05)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11, fontFamily: 'JetBrains Mono' }} />
                  <PolarRadiusAxis tick={false} axisLine={false} />
                  <Radar name="Traditional ML" dataKey="Traditional" stroke="#666" fill="#666" fillOpacity={0.1} strokeWidth={1} />
                  <Radar name="ProteinAI" dataKey="AI" stroke="#00f5ff" fill="#00f5ff" fillOpacity={0.15} strokeWidth={2} />
                  <Tooltip contentStyle={{ background: '#0d1530', border: '1px solid rgba(0,245,255,0.2)', borderRadius: '8px', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

// ===== FEATURES SECTION =====
const Features = () => {
  useReveal()

  const features = [
    {
      icon: <Atom size={32} />,
      title: 'Protein Structure Analysis',
      desc: 'Deep parsing of 3D protein topology from PDB files, AlphaFold models, and cryo-EM data with sub-angstrom resolution.',
      color: '#00f5ff',
      badge: 'Core Engine',
    },
    {
      icon: <BarChart2 size={32} />,
      title: 'Drugability Scoring',
      desc: 'Multi-factor scoring across hydrophobicity, pocket geometry, electrostatic potential, and evolutionary conservation.',
      color: '#8b5cf6',
      badge: 'Proprietary',
    },
    {
      icon: <Search size={32} />,
      title: 'Binding Pocket Detection',
      desc: 'AI-powered cavity identification using geometric clustering + energy minimization with allosteric site prediction.',
      color: '#10b981',
      badge: 'Patent Pending',
    },
    {
      icon: <Brain size={32} />,
      title: 'Molecular Reasoning Engine',
      desc: 'LLM-guided reasoning over protein-ligand interaction graphs, incorporating domain knowledge and literature.',
      color: '#f0abfc',
      badge: 'Cutting Edge',
    },
    {
      icon: <Eye size={32} />,
      title: 'Explainable AI',
      desc: 'Full interpretability stack with attention visualization, feature attribution, and natural language explanations for every prediction.',
      color: '#fbbf24',
      badge: 'Transparent',
    },
    {
      icon: <Bot size={32} />,
      title: 'Autonomous Research Agents',
      desc: 'Specialized AI agents that autonomously search literature, propose experiments, and refine hypotheses in real time.',
      color: '#0080ff',
      badge: 'Agentic',
    },
    {
      icon: <Target size={32} />,
      title: 'Predictive Drug Targeting',
      desc: 'End-to-end candidate prioritization from target selection to ADMET profiling using multi-objective optimization.',
      color: '#ff6b6b',
      badge: 'Full Pipeline',
    },
    {
      icon: <Layers size={32} />,
      title: 'Structural Visualization',
      desc: 'Interactive 3D molecular visualization with electrostatic surface mapping, cavity highlighting, and comparative analysis.',
      color: '#34d399',
      badge: 'Interactive',
    },
  ]

  return (
    <section id="research" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-25" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 80% 30%, rgba(0,245,255,0.04) 0%, transparent 60%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// PLATFORM CAPABILITIES</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Built for <span className="gradient-text-secondary">Scientific Precision</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            Eight specialized modules working in concert to deliver the most comprehensive protein analysis platform ever built.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {features.map((f, i) => (
            <div key={i} className="glass-card feature-card gradient-border p-6 rounded-2xl reveal"
              style={{ transitionDelay: `${i * 0.05}s` }}>
              {/* Icon */}
              <div className="relative mb-5">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                  style={{ background: `${f.color}12`, border: `1px solid ${f.color}25` }}>
                  <span style={{ color: f.color }}>{f.icon}</span>
                </div>
                <div className="absolute -top-1 -right-1 px-2 py-0.5 rounded-full text-xs font-mono"
                  style={{ background: `${f.color}15`, border: `1px solid ${f.color}30`, color: f.color }}>
                  {f.badge}
                </div>
              </div>

              <h3 className="font-display font-bold text-white text-sm mb-2 leading-tight">{f.title}</h3>
              <p className="font-body text-xs text-gray-500 leading-relaxed">{f.desc}</p>

              {/* Hover indicator */}
              <div className="mt-4 flex items-center gap-1 group cursor-pointer">
                <span className="font-mono text-xs" style={{ color: f.color }}>Learn more</span>
                <ArrowRight size={12} style={{ color: f.color }} className="group-hover:translate-x-1 transition-transform" />
              </div>

              {/* Bottom glow line */}
              <div className="absolute bottom-0 left-0 right-0 h-px rounded-full"
                style={{ background: `linear-gradient(90deg, transparent, ${f.color}40, transparent)` }} />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ===== SCIENTIFIC VISUALIZATION =====
const ScientificViz = () => {
  useReveal()

  const drugabilityData = [
    { protein: 'BCL-2', score: 94, binding: 87, novelty: 76 },
    { protein: 'KRAS G12C', score: 88, binding: 91, novelty: 94 },
    { protein: 'CDK4/6', score: 82, binding: 78, novelty: 65 },
    { protein: 'EGFR T790M', score: 96, binding: 93, novelty: 71 },
    { protein: 'PIK3CA', score: 79, binding: 82, novelty: 88 },
    { protein: 'BRAF V600E', score: 91, binding: 89, novelty: 73 },
  ]

  const confidenceTimeline = Array.from({ length: 20 }, (_, i) => ({
    epoch: i,
    confidence: 45 + Math.log(i + 1) * 20 + Math.sin(i) * 3,
    baseline: 45 + i * 0.5,
  }))

  const molecularNetwork = [
    { name: 'Target A', connections: 8, weight: 94 },
    { name: 'Target B', connections: 6, weight: 78 },
    { name: 'Target C', connections: 11, weight: 91 },
    { name: 'Target D', connections: 4, weight: 65 },
    { name: 'Target E', connections: 9, weight: 87 },
    { name: 'Target F', connections: 7, weight: 72 },
  ]

  return (
    <section className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-20" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 50% 80%, rgba(16,185,129,0.05) 0%, transparent 60%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// SCIENTIFIC DASHBOARD</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Real-Time <span className="gradient-text-secondary">Molecular Intelligence</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            Bloomberg terminal meets biotech AI lab. Live analysis, interactive visualizations, publication-ready outputs.
          </p>
        </div>

        {/* Top charts row */}
        <div className="grid lg:grid-cols-3 gap-6 mb-6">
          {/* Drugability bar chart */}
          <div className="lg:col-span-2 glass-card p-6 rounded-2xl reveal-left">
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="section-label mb-1">PROTEIN ANALYSIS BATCH #4471</div>
                <div className="font-display font-bold text-white">Multi-Factor Drugability Scores</div>
              </div>
              <div className="flex gap-2">
                {['Drugability','Binding','Novelty'].map((l, i) => (
                  <div key={i} className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full" style={{ background: ['#00f5ff','#8b5cf6','#10b981'][i] }} />
                    <span className="font-mono text-xs text-gray-400">{l}</span>
                  </div>
                ))}
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={drugabilityData} barSize={10} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
                <XAxis dataKey="protein" stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.5)' }} />
                <YAxis domain={[50, 100]} stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.5)' }} />
                <Tooltip contentStyle={{ background: '#0d1530', border: '1px solid rgba(0,245,255,0.2)', borderRadius: '8px', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Bar dataKey="score" fill="#00f5ff" radius={[2,2,0,0]}>
                  {drugabilityData.map((_, i) => (
                    <Cell key={i} fill={`rgba(0,245,255,${0.5 + i*0.08})`} />
                  ))}
                </Bar>
                <Bar dataKey="binding" fill="#8b5cf6" radius={[2,2,0,0]}>
                  {drugabilityData.map((_, i) => (
                    <Cell key={i} fill={`rgba(139,92,246,${0.5 + i*0.07})`} />
                  ))}
                </Bar>
                <Bar dataKey="novelty" fill="#10b981" radius={[2,2,0,0]}>
                  {drugabilityData.map((_, i) => (
                    <Cell key={i} fill={`rgba(16,185,129,${0.5 + i*0.08})`} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Live status panel */}
          <div className="glass-card p-6 rounded-2xl reveal-right">
            <div className="section-label mb-4">LIVE ANALYSIS STATUS</div>
            <div className="space-y-4">
              {[
                { label: 'Structure Parsing', progress: 100, status: 'Complete', color: '#10b981' },
                { label: 'Pocket Detection', progress: 100, status: 'Complete', color: '#10b981' },
                { label: 'Binding Analysis', progress: 87, status: 'Running', color: '#00f5ff' },
                { label: 'ADMET Profiling', progress: 43, status: 'Queued', color: '#8b5cf6' },
                { label: 'Report Generation', progress: 0, status: 'Pending', color: '#666' },
              ].map((task, i) => (
                <div key={i} className="p-3 rounded-xl" style={{ background: `${task.color}08`, border: `1px solid ${task.color}20` }}>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-body text-sm text-gray-300">{task.label}</span>
                    <span className="font-mono text-xs" style={{ color: task.color }}>{task.status}</span>
                  </div>
                  <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full transition-all duration-1000"
                      style={{ width: `${task.progress}%`, background: task.color,
                        boxShadow: task.progress > 0 ? `0 0 8px ${task.color}` : 'none' }} />
                  </div>
                  <div className="font-mono text-xs text-gray-500 mt-1">{task.progress}% complete</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom charts row */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* AI confidence over training */}
          <div className="glass-card p-6 rounded-2xl reveal-left">
            <div className="section-label mb-1">MODEL CONVERGENCE</div>
            <div className="font-display font-bold text-white mb-4">AI Confidence Over Reasoning Epochs</div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={confidenceTimeline}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="epoch" stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.4)' }}
                  label={{ value: 'Reasoning Depth', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
                <YAxis domain={[40, 100]} stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.4)' }} />
                <Tooltip contentStyle={{ background: '#0d1530', border: '1px solid rgba(0,245,255,0.2)', borderRadius: '8px', fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Line type="monotone" dataKey="baseline" stroke="rgba(255,255,255,0.15)" strokeWidth={1} dot={false} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="confidence" stroke="#00f5ff" strokeWidth={2} dot={false}
                  filter="drop-shadow(0 0 4px rgba(0,245,255,0.6))" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Scatter: molecular network */}
          <div className="glass-card p-6 rounded-2xl reveal-right">
            <div className="section-label mb-1">MOLECULAR NETWORK MAP</div>
            <div className="font-display font-bold text-white mb-4">Target Connectivity vs Drugability Weight</div>
            <ResponsiveContainer width="100%" height={200}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="connections" name="Connections" stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.4)' }}
                  label={{ value: 'Network Connections', position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.3)', fontSize: 10 }} />
                <YAxis dataKey="weight" name="Drugability" domain={[60, 100]} stroke="rgba(255,255,255,0.2)"
                  tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: 'rgba(255,255,255,0.4)' }} />
                <Tooltip
                  cursor={{ fill: 'rgba(0,245,255,0.05)' }}
                  contentStyle={{ background: '#0d1530', border: '1px solid rgba(0,245,255,0.2)', borderRadius: '8px', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                  formatter={(val, name) => [val, name === 'weight' ? 'Drugability %' : name]}
                />
                <Scatter data={molecularNetwork} fill="#00f5ff" fillOpacity={0.7}
                  filter="drop-shadow(0 0 4px rgba(0,245,255,0.5))">
                  {molecularNetwork.map((_, i) => (
                    <Cell key={i} fill={['#00f5ff','#8b5cf6','#10b981','#f0abfc','#fbbf24','#0080ff'][i]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  )
}

// ===== HOW IT WORKS =====
const HowItWorks = () => {
  useReveal()

  const steps = [
    {
      num: '01',
      title: 'Protein Input',
      desc: 'Upload PDB files, FASTA sequences, or AlphaFold predictions. Supports 50+ structural formats.',
      icon: <FlaskConical size={24} />,
      color: '#00f5ff',
    },
    {
      num: '02',
      title: 'Structural Parsing',
      desc: 'Deep topology analysis extracts secondary structures, domains, and functionally relevant regions.',
      icon: <Layers size={24} />,
      color: '#0080ff',
    },
    {
      num: '03',
      title: 'AI Agent Reasoning',
      desc: 'Six specialized agents analyze the protein in parallel — structure, pocket, chemistry, pathway, drug, safety.',
      icon: <Brain size={24} />,
      color: '#8b5cf6',
    },
    {
      num: '04',
      title: 'Drugability Prediction',
      desc: 'Proprietary scoring engine integrates 140+ molecular features to compute a comprehensive drugability index.',
      icon: <Sigma size={24} />,
      color: '#f0abfc',
    },
    {
      num: '05',
      title: 'Molecular Insights',
      desc: 'Detailed binding analysis, ADMET profiling, resistance mutation mapping, and selectivity scoring.',
      icon: <Microscope size={24} />,
      color: '#10b981',
    },
    {
      num: '06',
      title: 'Research Recommendations',
      desc: 'Actionable report with top drug candidates, synthesis pathways, experimental validations, and literature citations.',
      icon: <Sparkles size={24} />,
      color: '#fbbf24',
    },
  ]

  return (
    <section id="pipeline" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-20" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 50% 50%, rgba(0,128,255,0.05) 0%, transparent 70%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// ANALYSIS PIPELINE</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            From Sequence to <span className="gradient-text-primary">Discovery</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            A six-stage autonomous pipeline that transforms raw protein data into actionable drug discovery intelligence.
          </p>
        </div>

        {/* Pipeline flow */}
        <div className="relative">
          {/* Connecting lines (desktop) */}
          <div className="hidden lg:block absolute top-[52px] left-0 right-0 h-px"
            style={{ background: 'linear-gradient(90deg, transparent 5%, rgba(0,245,255,0.2) 20%, rgba(139,92,246,0.3) 50%, rgba(16,185,129,0.2) 80%, transparent 95%)' }} />

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {steps.map((step, i) => (
              <div key={i} className="glass-card feature-card gradient-border p-6 rounded-2xl reveal relative"
                style={{ transitionDelay: `${i * 0.1}s` }}>
                {/* Step number badge */}
                <div className="flex items-center justify-between mb-5">
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                    style={{ background: `${step.color}15`, border: `1px solid ${step.color}30` }}>
                    <span style={{ color: step.color }}>{step.icon}</span>
                  </div>
                  <div className="font-display font-black text-4xl opacity-10 text-white">{step.num}</div>
                </div>

                <h3 className="font-display font-bold text-white text-base mb-2">{step.title}</h3>
                <p className="font-body text-sm text-gray-500 leading-relaxed">{step.desc}</p>

                {/* Arrow */}
                {i < steps.length - 1 && (
                  <div className="absolute -right-3 top-1/2 -translate-y-1/2 hidden lg:flex w-6 h-6 items-center justify-center z-10">
                    <ChevronRight size={16} style={{ color: step.color }} />
                  </div>
                )}

                {/* Bottom accent */}
                <div className="absolute bottom-0 left-0 w-full h-0.5 rounded-b-2xl"
                  style={{ background: `linear-gradient(90deg, ${step.color}40, transparent)` }} />
              </div>
            ))}
          </div>
        </div>

        {/* SVG pipeline flow diagram */}
        <div className="mt-12 glass-card p-6 rounded-2xl reveal overflow-x-auto">
          <div className="section-label mb-4 text-center">AGENT COMMUNICATION PROTOCOL</div>
          <svg viewBox="0 0 900 100" className="w-full min-w-[600px]" style={{ height: '100px' }}>
            <defs>
              <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill="rgba(0,245,255,0.6)" />
              </marker>
            </defs>
            {/* Node positions */}
            {[
              [75,'Input','#00f5ff'],[225,'Parse','#0080ff'],[375,'Reason','#8b5cf6'],
              [525,'Score','#f0abfc'],[675,'Analyze','#10b981'],[825,'Output','#fbbf24']
            ].map(([x, label, color], i) => (
              <g key={i}>
                <circle cx={x} cy={50} r={28} fill={`${color}15`} stroke={color} strokeWidth={1} strokeOpacity={0.5} />
                <circle cx={x} cy={50} r={34} fill="none" stroke={color} strokeWidth={0.5} strokeOpacity={0.2}
                  style={{ animation: `ring-rotate ${4 + i}s linear infinite` }} />
                <text x={x} y={55} fill={color} textAnchor="middle" fontSize="11"
                  fontFamily="Orbitron" fontWeight="700">{label}</text>
                {i < 5 && (
                  <line x1={x + 30} y1={50} x2={x + 120} y2={50}
                    stroke="rgba(0,245,255,0.3)" strokeWidth={1}
                    strokeDasharray="5 3" markerEnd="url(#arrow)"
                    style={{ animation: `flow 1.5s linear ${i * 0.3}s infinite` }} />
                )}
              </g>
            ))}
          </svg>
        </div>
      </div>
    </section>
  )
}

// ===== IMPACT SECTION =====
const Impact = () => {
  useReveal()

  const { count: c1, ref: r1 } = useCounter(94)
  const { count: c2, ref: r2 } = useCounter(100)
  const { count: c3, ref: r3 } = useCounter(2400000)
  const { count: c4, ref: r4 } = useCounter(87)

  const metrics = [
    { ref: r1, count: c1, suffix: '%', label: 'Prediction Accuracy', sub: 'vs 45% traditional ML', color: '#00f5ff', icon: <Target size={20} /> },
    { ref: r2, count: c2, suffix: 'x', label: 'Faster Discovery', sub: 'Average time compression', color: '#10b981', icon: <Zap size={20} /> },
    { ref: r3, count: c3, suffix: '+', label: 'Proteins Analyzed', sub: 'Across all PDB entries', color: '#8b5cf6', icon: <Dna size={20} /> },
    { ref: r4, count: c4, suffix: '%', label: 'Cost Reduction', sub: 'In pre-clinical R&D spend', color: '#f0abfc', icon: <DollarSign size={20} /> },
  ]

  return (
    <section id="impact" className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-25" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 30% 50%, rgba(16,185,129,0.06) 0%, transparent 60%)' }} />
      <Particles count={15} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// MEASURABLE IMPACT</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            Numbers That <span className="gradient-text-secondary">Redefine the Field</span>
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            Validated across 200+ research institutions, 15 pharmaceutical companies, and 3 ongoing clinical trials.
          </p>
        </div>

        {/* Big metrics */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {metrics.map((m, i) => (
            <div key={i} ref={m.ref} className="glass-card p-6 rounded-2xl text-center reveal gradient-border"
              style={{ transitionDelay: `${i * 0.1}s` }}>
              <div className="flex justify-center mb-3">
                <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                  style={{ background: `${m.color}15`, border: `1px solid ${m.color}25` }}>
                  <span style={{ color: m.color }}>{m.icon}</span>
                </div>
              </div>
              <div className="font-display font-black text-4xl md:text-5xl mb-1" style={{ color: m.color }}>
                {m.count > 999 ? `${(m.count / 1000000).toFixed(1)}M` : m.count}
                {m.suffix}
              </div>
              <div className="font-display font-bold text-white text-sm mb-1">{m.label}</div>
              <div className="font-mono text-xs text-gray-500">{m.sub}</div>
            </div>
          ))}
        </div>

        {/* Impact areas */}
        <div className="grid lg:grid-cols-3 gap-6">
          {[
            {
              title: 'Oncology',
              desc: 'Identified 23 novel cancer targets across KRAS, BCL-2, and CDK family with unprecedented precision.',
              projects: 47,
              color: '#00f5ff',
              icon: <Target size={28} />,
            },
            {
              title: 'Neurology',
              desc: 'Mapped 156 druggable pockets in neurodegenerative disease proteins including tau and alpha-synuclein.',
              projects: 31,
              color: '#8b5cf6',
              icon: <Brain size={28} />,
            },
            {
              title: 'Infectious Disease',
              desc: 'Accelerated antiviral discovery against 12 novel pathogen proteases including emerging variants.',
              projects: 29,
              color: '#10b981',
              icon: <Shield size={28} />,
            },
          ].map((area, i) => (
            <div key={i} className="glass-card feature-card p-7 rounded-2xl reveal"
              style={{ transitionDelay: `${i * 0.15}s` }}>
              <div className="flex items-start gap-4 mb-4">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0"
                  style={{ background: `${area.color}12`, border: `1px solid ${area.color}25` }}>
                  <span style={{ color: area.color }}>{area.icon}</span>
                </div>
                <div>
                  <h3 className="font-display font-black text-white text-xl">{area.title}</h3>
                  <div className="font-mono text-xs" style={{ color: area.color }}>{area.projects} active projects</div>
                </div>
              </div>
              <p className="font-body text-sm text-gray-400 leading-relaxed">{area.desc}</p>
              <div className="mt-4 pt-4 border-t border-white/5 flex items-center gap-2">
                <div className="w-2 h-2 rounded-full pulse-dot" style={{ background: area.color }} />
                <span className="font-mono text-xs" style={{ color: area.color }}>Real-world validated</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ===== RESEARCH SECTION =====
const Research = () => {
  useReveal()

  const papers = [
    {
      id: 'PAI-2024-001',
      title: 'Agentic Multi-Step Reasoning for Protein-Ligand Interaction Prediction',
      authors: 'Chen, R. • Patel, S. • Nakamura, H. • Williams, K.',
      journal: 'Nature Computational Science',
      year: '2024',
      citations: 847,
      impact: 12.4,
      tags: ['Agentic AI', 'Drug Discovery', 'Transformer'],
      color: '#00f5ff',
    },
    {
      id: 'PAI-2024-002',
      title: 'Hierarchical Pocket Detection via Geometric Deep Learning on Protein Graphs',
      authors: 'Singh, A. • Rodriguez, M. • Kim, J.',
      journal: 'Cell Systems',
      year: '2024',
      citations: 523,
      impact: 9.8,
      tags: ['Graph Neural Networks', 'Pocket Detection', 'Geometry'],
      color: '#8b5cf6',
    },
    {
      id: 'PAI-2023-018',
      title: 'Reinforcement Learning for Adaptive Molecular Optimization in Lead Generation',
      authors: 'Zhang, L. • Okonkwo, F. • Braun, T.',
      journal: 'Journal of Chemical Information',
      year: '2023',
      citations: 1204,
      impact: 14.1,
      tags: ['Reinforcement Learning', 'Lead Optimization', 'ADMET'],
      color: '#10b981',
    },
  ]

  const architecture = [
    { layer: 'Input Layer', desc: 'PDB / FASTA / SMILES', nodes: 512, color: '#00f5ff' },
    { layer: 'Structural Encoder', desc: 'SE(3)-Equivariant GNN', nodes: 1024, color: '#0080ff' },
    { layer: 'Attention Pooling', desc: 'Multi-head cross-attention', nodes: 768, color: '#8b5cf6' },
    { layer: 'Reasoning Module', desc: 'Agentic CoT + RAG', nodes: 2048, color: '#f0abfc' },
    { layer: 'Prediction Head', desc: 'Multi-task classifier', nodes: 256, color: '#10b981' },
    { layer: 'Output Layer', desc: 'Drugability + Insights', nodes: 128, color: '#fbbf24' },
  ]

  return (
    <section className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-20" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 70% 20%, rgba(240,171,252,0.04) 0%, transparent 60%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// RESEARCH & PUBLICATIONS</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            <span className="gradient-text-warm">Peer-Reviewed</span> Science
          </h2>
          <p className="font-body text-lg text-gray-400 max-w-2xl mx-auto">
            Built on rigorous research. Every model, every prediction, every insight backed by published, reproducible science.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Research papers */}
          <div className="space-y-4 reveal-left">
            <div className="section-label mb-3">FEATURED PUBLICATIONS</div>
            {papers.map((paper, i) => (
              <div key={i} className="glass-card p-5 rounded-2xl feature-card gradient-border">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <div className="font-mono text-xs" style={{ color: paper.color }}>{paper.id}</div>
                  <div className="flex gap-1">
                    {paper.tags.slice(0, 2).map((tag, j) => (
                      <span key={j} className="font-mono text-xs px-2 py-0.5 rounded"
                        style={{ background: `${paper.color}15`, color: paper.color, border: `1px solid ${paper.color}25` }}>
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                <h4 className="font-display font-bold text-white text-sm leading-tight mb-2">{paper.title}</h4>
                <div className="font-mono text-xs text-gray-500 mb-3">{paper.authors}</div>
                <div className="flex items-center justify-between text-xs">
                  <span className="font-body text-gray-400 italic">{paper.journal} · {paper.year}</span>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-gray-400">{paper.citations.toLocaleString()} citations</span>
                    <span className="font-mono" style={{ color: paper.color }}>IF {paper.impact}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Architecture diagram */}
          <div className="glass-card p-6 rounded-2xl reveal-right">
            <div className="section-label mb-4">MODEL ARCHITECTURE</div>
            <div className="font-display font-bold text-white mb-6">ProteinAI Neural Stack v3.2</div>
            <div className="space-y-3">
              {architecture.map((layer, i) => (
                <div key={i} className="flex items-center gap-3 p-3 rounded-xl"
                  style={{ background: `${layer.color}08`, border: `1px solid ${layer.color}15` }}>
                  <div className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: `${layer.color}20` }}>
                    <span className="font-display font-black text-xs" style={{ color: layer.color }}>{i + 1}</span>
                  </div>
                  <div className="flex-1">
                    <div className="font-display font-bold text-white text-xs">{layer.layer}</div>
                    <div className="font-mono text-xs text-gray-500">{layer.desc}</div>
                  </div>
                  <div className="font-mono text-xs" style={{ color: layer.color }}>{layer.nodes.toLocaleString()}d</div>
                  {i < architecture.length - 1 && (
                    <div className="absolute left-[38px] mt-12 w-px h-4"
                      style={{ background: layer.color, opacity: 0.3 }} />
                  )}
                </div>
              ))}
            </div>
            <div className="mt-4 p-3 rounded-xl bg-cyan-400/5 border border-cyan-400/20">
              <div className="flex items-center gap-2">
                <Cpu size={14} className="text-cyan-400" />
                <span className="font-mono text-xs text-cyan-400">Trained on 500M protein-ligand pairs · 2.1B parameters</span>
              </div>
            </div>
          </div>
        </div>

        {/* Institutions */}
        <div className="glass-card p-6 rounded-2xl reveal text-center">
          <div className="section-label mb-4">RESEARCH PARTNERSHIPS</div>
          <div className="flex flex-wrap justify-center gap-6 opacity-50">
            {['Stanford Medicine', 'MIT CSAIL', 'Broad Institute', 'DeepMind Health', 'Pfizer R&D', 'UCSF QB3', 'Weill Cornell', 'Johns Hopkins'].map((inst, i) => (
              <span key={i} className="font-display font-bold text-white text-sm tracking-wider">{inst}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// ===== TESTIMONIALS =====
const Testimonials = () => {
  useReveal()

  const testimonials = [
    {
      quote: "ProteinAI identified a BCL-2 binding pocket we had missed for three years. Within 48 hours, we had a lead compound that passed pre-clinical validation. This platform is a paradigm shift.",
      name: "Dr. Sarah Nakamura",
      role: "Director of Computational Drug Discovery",
      org: "Genexis Therapeutics",
      avatar: 'SN',
      color: '#00f5ff',
    },
    {
      quote: "The agentic reasoning layer is unlike anything in the field. It doesn't just predict — it explains its reasoning in a way that actually teaches our scientists new biology. Remarkable.",
      name: "Prof. Marcus Osei",
      role: "Chair, Structural Biology",
      org: "MIT Department of Biology",
      avatar: 'MO',
      color: '#8b5cf6',
    },
    {
      quote: "We compressed an 18-month target validation campaign into 11 days. The multi-agent architecture handles complexity that would take a team of 40 researchers to address. ROI is extraordinary.",
      name: "Dr. Priya Anantharaman",
      role: "VP Drug Discovery",
      org: "NovaBio Sciences",
      avatar: 'PA',
      color: '#10b981',
    },
  ]

  return (
    <section className="relative py-32 overflow-hidden">
      <div className="absolute inset-0 grid-bg opacity-15" />
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 50% 100%, rgba(139,92,246,0.08) 0%, transparent 60%)' }} />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <div className="text-center mb-16 reveal">
          <div className="section-label mb-4">// FIELD VOICES</div>
          <h2 className="font-display font-black text-4xl md:text-5xl text-white mb-4">
            What <span className="gradient-text-primary">Scientists</span> Are Saying
          </h2>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          {testimonials.map((t, i) => (
            <div key={i} className="glass-card feature-card gradient-border p-7 rounded-2xl reveal"
              style={{ transitionDelay: `${i * 0.15}s` }}>
              {/* Quote marks */}
              <div className="font-display text-6xl leading-none mb-4 opacity-20" style={{ color: t.color }}>"</div>

              <p className="font-body text-base text-gray-300 leading-relaxed mb-6 italic">
                "{t.quote}"
              </p>

              <div className="flex items-center gap-3 pt-4 border-t border-white/5">
                <div className="w-10 h-10 rounded-full flex items-center justify-center font-display font-black text-sm"
                  style={{ background: `${t.color}20`, color: t.color, border: `1px solid ${t.color}30` }}>
                  {t.avatar}
                </div>
                <div>
                  <div className="font-display font-bold text-white text-sm">{t.name}</div>
                  <div className="font-mono text-xs text-gray-500">{t.role}</div>
                  <div className="font-mono text-xs" style={{ color: t.color }}>{t.org}</div>
                </div>
              </div>

              {/* Stars */}
              <div className="flex gap-1 mt-3">
                {Array.from({ length: 5 }).map((_, j) => (
                  <Star key={j} size={12} fill={t.color} style={{ color: t.color }} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ===== FINAL CTA =====
const CTA = ({ onOpenConsole }) => {
  return (
    <section className="relative py-40 overflow-hidden">
      {/* Dramatic background */}
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse at 50% 50%, rgba(0,128,255,0.12) 0%, rgba(139,92,246,0.08) 40%, transparent 70%)' }} />
      <div className="absolute inset-0 grid-bg opacity-30" />
      <Particles count={40} />

      {/* Top border glow */}
      <div className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, #00f5ff, #8b5cf6, transparent)' }} />

      <div className="relative z-10 max-w-4xl mx-auto px-6 text-center">
        {/* Badge */}
        <div className="inline-flex items-center gap-2 glass-card border-glow-cyan px-4 py-2 rounded-full mb-10">
          <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
          <span className="section-label">Platform Access Now Open</span>
        </div>

        {/* Headline */}
        <h2 className="font-display font-black text-5xl md:text-6xl lg:text-7xl text-white leading-tight mb-6">
          The Future of Therapeutic
          <span className="block gradient-text-primary">Discovery is Autonomous.</span>
        </h2>

        <p className="font-body text-xl text-gray-400 leading-relaxed mb-12 max-w-2xl mx-auto">
          Join 200+ research institutions already using ProteinAI to accelerate drug discovery. The next breakthrough molecule might be one analysis away.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-wrap justify-center gap-4 mb-16">
          <button className="btn-primary flex items-center gap-2 text-sm px-8 py-4" onClick={onOpenConsole}>
            <Zap size={16} />
            Get Started Free
          </button>
          <button className="btn-secondary flex items-center gap-2 text-sm px-8 py-4" onClick={onOpenConsole}>
            <Play size={16} />
            Explore Platform
          </button>
          <a href="#research" className="flex items-center gap-2 text-sm px-8 py-4 font-display font-bold tracking-widest uppercase text-gray-400 hover:text-white transition-colors border border-white/10 rounded hover:border-white/20">
            <ExternalLink size={14} />
            Read Research
          </a>
        </div>

        {/* Trust signals */}
        <div className="flex flex-wrap justify-center gap-8 text-center">
          {[
            { value: '200+', label: 'Institutions' },
            { value: '2.4M+', label: 'Proteins Analyzed' },
            { value: '94.7%', label: 'Accuracy' },
            { value: 'SOC2', label: 'Certified' },
          ].map((item, i) => (
            <div key={i}>
              <div className="font-display font-black text-2xl gradient-text-primary">{item.value}</div>
              <div className="font-mono text-xs text-gray-500 mt-0.5">{item.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

// ===== FOOTER =====
const Footer = () => (
  <footer className="relative border-t border-white/5 py-12 overflow-hidden">
    <div className="absolute inset-0 grid-bg opacity-10" />
    <div className="relative z-10 max-w-7xl mx-auto px-6">
      <div className="grid md:grid-cols-4 gap-8 mb-10">
        {/* Logo + tagline */}
        <div className="md:col-span-1">
          <div className="flex items-center gap-2 mb-3">
            <Dna size={20} className="text-cyan-400" />
            <span className="font-display font-bold text-white">PROTEIN<span style={{ color: '#00f5ff' }}>AI</span></span>
          </div>
          <p className="font-body text-sm text-gray-500 leading-relaxed">
            Autonomous intelligence for the future of drug discovery.
          </p>
          <div className="flex gap-3 mt-4">
            {['Twitter', 'GitHub', 'LinkedIn'].map((s) => (
              <div key={s} className="w-8 h-8 rounded glass-card flex items-center justify-center cursor-pointer hover:border-cyan-400/30 transition-colors">
                <Globe size={12} className="text-gray-500" />
              </div>
            ))}
          </div>
        </div>

        {/* Links */}
        {[
          { title: 'Platform', links: ['Dashboard', 'API Docs', 'Integrations', 'Changelog'] },
          { title: 'Research', links: ['Publications', 'Benchmarks', 'Datasets', 'Open Source'] },
          { title: 'Company', links: ['About', 'Careers', 'Blog', 'Contact'] },
        ].map((col) => (
          <div key={col.title}>
            <div className="font-display font-bold text-white text-sm mb-3 tracking-widest">{col.title}</div>
            <div className="space-y-2">
              {col.links.map((link) => (
                <div key={link} className="font-body text-sm text-gray-500 hover:text-gray-300 cursor-pointer transition-colors">{link}</div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="border-t border-white/5 pt-6 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="font-mono text-xs text-gray-600">
          © 2026 ProteinAI. All rights reserved. Built for science.
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-gray-600">Privacy Policy</span>
          <span className="font-mono text-xs text-gray-600">Terms of Service</span>
          <div className="flex items-center gap-1">
            <Lock size={10} className="text-emerald-400" />
            <span className="font-mono text-xs text-emerald-400">SOC 2 Type II</span>
          </div>
        </div>
      </div>
    </div>
  </footer>
)

// ===== ANALYSIS CONSOLE =====
const AnalysisConsole = ({ onBack }) => {
  const [form, setForm] = useState({
    protein_input: '2ITX',
    gene_symbol: 'EGFR',
    disease_name: 'breast cancer',
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const presets = [
    { label: 'EGFR / breast cancer', protein_input: '2ITX', gene_symbol: 'EGFR', disease_name: 'breast cancer' },
    { label: 'HER2 / breast cancer', protein_input: '3PP0', gene_symbol: 'ERBB2', disease_name: 'breast cancer' },
    { label: 'BCL2 / lymphoma', protein_input: '4LVT', gene_symbol: 'BCL2', disease_name: 'lymphoma' },
  ]

  const updateField = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }))
  }

  const applyPreset = (preset) => {
    setForm({
      protein_input: preset.protein_input,
      gene_symbol: preset.gene_symbol,
      disease_name: preset.disease_name,
    })
  }

  const printReport = () => {
    window.print()
  }

  const runAnalysis = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })

      if (!response.ok) {
        throw new Error(`Analysis failed with status ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const summaryCards = result?.summary
    ? [
        { label: 'Structure', value: result.summary.structure.score, detail: result.summary.structure.assessment, color: '#00f5ff' },
        { label: 'Biology', value: result.summary.biology.score, detail: result.summary.biology.verdict, color: '#10b981' },
        { label: 'Safety', value: result.summary.safety.score, detail: result.summary.safety.verdict, color: '#f0abfc' },
        { label: 'Chemistry', value: result.summary.chemistry.score, detail: result.summary.chemistry.verdict, color: '#8b5cf6' },
      ]
    : []

  const contributionData = result?.decision?.components
    ? Object.entries(result.decision.components).map(([name, value]) => ({
        name: name[0].toUpperCase() + name.slice(1),
        value,
      }))
    : []

  const radarData = summaryCards.map((card) => ({
    subject: card.label,
    score: Number(card.value || 0),
    fullMark: 1,
  }))

  const biologyEvidenceData = result?.agents?.biology?.evidence
    ? Object.entries(result.agents.biology.evidence).map(([name, value]) => ({
        name,
        value,
      }))
    : []

  const safetyData = result?.agents?.safety?.components
    ? Object.entries(result.agents.safety.components)
        .filter(([, value]) => value !== null)
        .map(([name, value]) => ({
          name: name[0].toUpperCase() + name.slice(1),
          value,
        }))
    : []

  const chemistryData = result?.agents?.chemistry?.sample_results
    ? result.agents.chemistry.sample_results.map((item) => ({
        name: item.molecule_chembl_id,
        score: item.score,
        binding: item.binding_score,
      }))
    : []

  const scorePercent = Math.round((result?.decision?.integrated_score || 0) * 100)
  const positives = result
    ? [
        ...(result.agents?.biology?.positive_signals || []),
        ...(result.agents?.chemistry?.warning_signals?.length ? [] : ['Usable ligand activity data available']),
      ]
    : []
  const warnings = result
    ? [
        ...(result.agents?.biology?.warning_signals || []),
        ...(result.agents?.safety?.warning_signals || []),
        ...(result.agents?.chemistry?.warning_signals || []),
      ]
    : []

  return (
    <main className="relative min-h-screen overflow-hidden py-10">
      <div className="absolute inset-0 grid-bg opacity-30" />
      <BlobBackground />

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <button className="inline-flex items-center gap-2 font-mono text-xs text-gray-400 hover:text-cyan-400 transition-colors mb-8"
          onClick={onBack}>
          <ArrowLeft size={14} />
          Back to platform
        </button>

        <div className="grid lg:grid-cols-[360px_1fr] gap-6 items-start">
          <section className="glass-card rounded-2xl p-6">
            <div className="section-label mb-3">// AGENT INPUT</div>
            <h1 className="font-display font-black text-2xl text-white mb-6">Run Target Analysis</h1>

            <div className="mb-5">
              <div className="login-label">Demo presets</div>
              <div className="preset-list">
                {presets.map((preset) => (
                  <button key={preset.label} type="button" className="preset-chip" onClick={() => applyPreset(preset)}>
                    {preset.label}
                  </button>
                ))}
              </div>
            </div>

            <form className="space-y-5" onSubmit={runAnalysis}>
              <label className="block">
                <span className="login-label">Protein / PDB ID</span>
                <span className="login-field">
                  <Dna size={16} />
                  <input value={form.protein_input} onChange={updateField('protein_input')} placeholder="2ITX" />
                </span>
              </label>

              <label className="block">
                <span className="login-label">Gene symbol</span>
                <span className="login-field">
                  <Atom size={16} />
                  <input value={form.gene_symbol} onChange={updateField('gene_symbol')} placeholder="EGFR" />
                </span>
              </label>

              <label className="block">
                <span className="login-label">Disease</span>
                <span className="login-field">
                  <Activity size={16} />
                  <input value={form.disease_name} onChange={updateField('disease_name')} placeholder="breast cancer" />
                </span>
              </label>

              <button type="submit" className="btn-primary w-full flex items-center justify-center gap-2 py-4"
                disabled={loading}>
                <Sparkles size={15} />
                {loading ? 'Running Agents...' : 'Run Analysis'}
              </button>
            </form>

            {result && (
              <button type="button" className="btn-secondary w-full mt-4 flex items-center justify-center gap-2 py-4"
                onClick={printReport}>
                <ExternalLink size={15} />
                Export Report
              </button>
            )}
          </section>

          <section className="space-y-6">
            <div className="grid xl:grid-cols-[1fr_280px] gap-6">
              <div className="glass-card rounded-2xl p-6">
                <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
                  <div>
                    <div className="section-label mb-3">// DECISION ENGINE</div>
                    <h2 className="font-display font-black text-3xl text-white">
                      {result?.decision?.final_verdict || 'Awaiting analysis'}
                    </h2>
                  </div>
                  <div className="text-left md:text-right">
                    <div className="font-mono text-xs text-gray-500">Integrated score</div>
                    <div className="font-display font-black text-4xl gradient-text-primary">
                      {result?.decision?.integrated_score ?? '--'}
                    </div>
                  </div>
                </div>

                <p className="font-body text-gray-400 mt-4">
                  {result?.decision?.recommendation || 'Submit a target to combine structure, biology, safety, and chemistry evidence.'}
                </p>

                <div className="mt-6 grid md:grid-cols-2 gap-4">
                  <div className="signal-panel">
                    <div className="section-label mb-3">Positive signals</div>
                    <div className="space-y-2">
                      {(positives.length ? positives : ['No signals yet']).map((item) => (
                        <div key={item} className="signal-row">
                          <CheckCircle2 size={14} className="text-emerald-400" />
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="signal-panel">
                    <div className="section-label mb-3">Warnings</div>
                    <div className="space-y-2">
                      {(warnings.length ? warnings : ['No active warnings']).map((item) => (
                        <div key={item} className="signal-row">
                          <XCircle size={14} className={warnings.length ? 'text-rose-300' : 'text-cyan-300'} />
                          <span>{item}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="glass-card rounded-2xl p-6 flex flex-col items-center justify-center">
                <div className="section-label mb-4">Confidence</div>
                <div className="score-ring" style={{ '--score': scorePercent }}>
                  <div className="score-ring-inner">
                    <span>{result ? scorePercent : '--'}</span>
                    <small>{result ? '%' : ''}</small>
                  </div>
                </div>
                <div className="font-mono text-xs text-gray-500 mt-4 text-center">
                  Weighted confidence from all four agents
                </div>
              </div>
            </div>

            {error && (
              <div className="glass-card rounded-2xl p-5 border border-red-400/30 text-red-300 font-mono text-sm">
                {error}
              </div>
            )}

            <div className="grid md:grid-cols-2 xl:grid-cols-4 gap-4">
              {summaryCards.map((card) => (
                <div key={card.label} className="glass-card rounded-2xl p-5">
                  <div className="section-label mb-2">{card.label}</div>
                  <div className="font-display font-black text-3xl mb-2" style={{ color: card.color }}>
                    {card.value ?? '--'}
                  </div>
                  <div className="font-body text-sm text-gray-400">{card.detail || 'No detail'}</div>
                </div>
              ))}
            </div>

            {result && (
              <div className="grid xl:grid-cols-2 gap-6">
                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Weighted contribution</div>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={contributionData}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="name" stroke="#6b7280" tickLine={false} axisLine={false} />
                      <YAxis stroke="#6b7280" tickLine={false} axisLine={false} domain={[0, 0.4]} />
                      <Tooltip contentStyle={{ background: '#08111f', border: '1px solid rgba(0,245,255,0.2)' }} />
                      <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                        {contributionData.map((entry, index) => (
                          <Cell key={entry.name} fill={['#00f5ff', '#10b981', '#f0abfc', '#8b5cf6'][index]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </article>

                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Agent profile</div>
                  <ResponsiveContainer width="100%" height={260}>
                    <RadarChart data={radarData}>
                      <PolarGrid stroke="rgba(255,255,255,0.08)" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 10 }} />
                      <Radar dataKey="score" stroke="#00f5ff" fill="#00f5ff" fillOpacity={0.25} />
                    </RadarChart>
                  </ResponsiveContainer>
                </article>
              </div>
            )}

            {result && (
              <div className="grid xl:grid-cols-3 gap-6">
                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Biology evidence</div>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={biologyEvidenceData} layout="vertical" margin={{ left: 20 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" horizontal={false} />
                      <XAxis type="number" domain={[0, 1]} stroke="#6b7280" tickLine={false} axisLine={false} />
                      <YAxis type="category" dataKey="name" width={150} stroke="#9ca3af" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#08111f', border: '1px solid rgba(0,245,255,0.2)' }} />
                      <Bar dataKey="value" fill="#10b981" radius={[0, 6, 6, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </article>

                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Safety breakdown</div>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={safetyData}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="name" stroke="#6b7280" tickLine={false} axisLine={false} />
                      <YAxis domain={[0, 1]} stroke="#6b7280" tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#08111f', border: '1px solid rgba(0,245,255,0.2)' }} />
                      <Bar dataKey="value" fill="#f0abfc" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </article>

                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Chemistry samples</div>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={chemistryData}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                      <XAxis dataKey="name" stroke="#6b7280" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                      <YAxis domain={[0, 1]} stroke="#6b7280" tickLine={false} axisLine={false} />
                      <Tooltip contentStyle={{ background: '#08111f', border: '1px solid rgba(0,245,255,0.2)' }} />
                      <Bar dataKey="score" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </article>
              </div>
            )}

            {result?.agents && (
              <div className="grid xl:grid-cols-2 gap-6">
                <article className="glass-card rounded-2xl p-6 xl:col-span-2">
                  <div className="section-label mb-4">3D protein visualization</div>
                  <iframe
                    title="Protein visualization"
                    className="visualization-frame"
                    src="/api/visualization"
                  />
                </article>

                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Structure findings</div>
                  <div className="detail-grid">
                    <div>
                      <span>Volume</span>
                      <strong>{result.agents.structure.volume}</strong>
                    </div>
                    <div>
                      <span>Depth</span>
                      <strong>{result.agents.structure.depth_3d}</strong>
                    </div>
                    <div>
                      <span>Residues</span>
                      <strong>{result.agents.structure.residue_count}</strong>
                    </div>
                    <div>
                      <span>Hydrophobicity</span>
                      <strong>{result.agents.structure.hydrophobic_content}</strong>
                    </div>
                  </div>
                </article>

                <article className="glass-card rounded-2xl p-6">
                  <div className="section-label mb-4">Trace output</div>
                  <pre className="analysis-json compact">{JSON.stringify(result.decision, null, 2)}</pre>
                </article>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  )
}

// ===== MAIN APP =====
export default function App() {
  const [view, setView] = useState('landing')

  useReveal()

  return (
    <div className="min-h-screen bg-[#030712] noise">
      <CustomCursor />
      {view === 'landing' ? (
        <>
          <ScrollProgress />
          <Navbar onOpenConsole={() => setView('analysis')} />

          <main>
            <Hero onOpenConsole={() => setView('analysis')} />
            <Problem />
            <AgenticAI />
            <Features />
            <ScientificViz />
            <HowItWorks />
            <Impact />
            <Research />
            <Testimonials />
            <CTA onOpenConsole={() => setView('analysis')} />
          </main>

          <Footer />
        </>
      ) : (
        <AnalysisConsole onBack={() => setView('landing')} />
      )}
    </div>
  )
}
