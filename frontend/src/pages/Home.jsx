import { useEffect, useState } from 'react'
import { getHealth } from '../api'

export default function Home() {
  const [status, setStatus] = useState('Checking...')
  const [error, setError] = useState('')

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await getHealth()
        setStatus(res.status === 'ok' ? 'Running normally' : 'Unknown')
      } catch {
        setError('Backend not reachable')
      }
    }
    checkHealth()
  }, [])

  return (
    <section className="home-layout">
      <div className="hero-card">
        <div className="eyebrow">A personal algorithm-based fashion curator</div>
        <h1>Get the attention of a personal fashion curator, all online and from the comfort of your home.</h1>
        <p className="lead">
          Curate learns from your saved preferences, your uploaded inspiration, and the results you choose to like or dislike.
        </p>
        <p className="lead secondary">
          Upload pictures as inspiration and select whether the result is something you like. The more you interact, the better the results you get.
        </p>

        <div className="hero-actions">
          <button className="primary-btn">Start with Profile</button>
          <button className="ghost-btn">Explore Search</button>
        </div>
      </div>

      <div className="home-side">
        <div className="info-card">
          <h3>How Curate works</h3>
          <ul className="info-list">
            <li>Build a personal style profile with preferred and avoided attributes.</li>
            <li>Upload inspiration images so the system can learn visual signals from your taste.</li>
            <li>Search naturally using phrases like black hoodie, clean streetwear, or smart casual.</li>
            <li>Use like and dislike feedback so recommendations improve over time.</li>
          </ul>
        </div>

        <div className="info-card dark">
          <h3>Application status</h3>
          {error ? <p>{error}</p> : <p>{status}</p>}
          <p className="muted-light">If this says running normally, the recommendation engine and saved profile system are available.</p>
        </div>
      </div>
    </section>
  )
}
