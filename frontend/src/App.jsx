import { useEffect, useMemo, useState } from 'react'
import { likeItem, dislikeItem } from './api'
import Home from './pages/Home'
import Profile from './pages/Profile'
import Search from './pages/Search'
import ForYou from './pages/ForYou'

const NAV_ITEMS = [
  { key: 'home', label: 'Home' },
  { key: 'profile', label: 'Profile' },
  { key: 'search', label: 'Search' },
  { key: 'for-you', label: 'For You' },
]

export default function App() {
  const [activePage, setActivePage] = useState('home')
  const [userId, setUserId] = useState('demo_user')
  const [status, setStatus] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    setStatus('Use the same user ID whenever you want Curate to remember your profile, uploaded inspiration, and feedback.')
  }, [])

  async function handleFeedbackSaved(itemId, action) {
    try {
      if (action === 'like') {
        await likeItem(userId, itemId)
      } else {
        await dislikeItem(userId, itemId)
      }
      setStatus(`Saved ${action}. Curate is updating your recommendation profile.`)
      setRefreshKey((value) => value + 1)
    } catch (error) {
      setStatus(error.message)
    }
  }

  const pageHint = useMemo(() => {
    if (activePage === 'home') return 'Explore what Curate does and check whether the application is running.'
    if (activePage === 'profile') return 'Shape your taste with saved preferences and uploaded inspiration.'
    if (activePage === 'search') return 'Search naturally and tell Curate what you like or dislike.'
    return 'See your personalized feed improve as you interact with the platform.'
  }, [activePage])

  return (
    <div className="site-shell">
      <header className="topbar">
        <div className="brand-wrap">
          <div className="brand-logo">C</div>
          <div>
            <div className="brand-title">Curate</div>
            <div className="brand-subtitle">Personal fashion, thoughtfully curated</div>
          </div>
        </div>

        <nav className="top-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              className={activePage === item.key ? 'top-nav-btn active' : 'top-nav-btn'}
              onClick={() => setActivePage(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <label className="user-id-box">
          <span>User ID</span>
          <input value={userId} onChange={(e) => setUserId(e.target.value)} />
        </label>
      </header>

      <div className="page-shell">
        <main className="page-content">
          {activePage === 'home' && <Home />}
          {activePage === 'profile' && (
            <Profile userId={userId} onStatus={setStatus} onProfileSaved={() => setRefreshKey((v) => v + 1)} />
          )}
          {activePage === 'search' && (
            <Search userId={userId} onStatus={setStatus} onFeedbackSaved={handleFeedbackSaved} />
          )}
          {activePage === 'for-you' && (
            <ForYou userId={userId} refreshKey={refreshKey} onStatus={setStatus} onFeedbackSaved={handleFeedbackSaved} />
          )}
        </main>

        <aside className="status-sidebar">
          <div className="status-card">
            <div className="eyebrow">Application status</div>
            <p>{status}</p>
          </div>

          <div className="status-card soft">
            <div className="eyebrow">Current page</div>
            <h3>{NAV_ITEMS.find((item) => item.key === activePage)?.label}</h3>
            <p>{pageHint}</p>
          </div>
        </aside>
      </div>
    </div>
  )
}
