import { useState } from 'react'
import { searchCatalog } from '../api'
import ItemCard from '../components/ItemCard'

export default function Search({ userId, onStatus, onFeedbackSaved }) {
  const [query, setQuery] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [feedbackBusy, setFeedbackBusy] = useState(false)

  async function handleSearch() {
    if (!query.trim()) {
      onStatus('Enter a search query first.')
      return
    }
    setLoading(true)
    try {
      const data = await searchCatalog(userId, query)
      setItems(Array.isArray(data) ? data : [])
      onStatus(`Loaded ${Array.isArray(data) ? data.length : 0} search results.`)
    } catch (error) {
      onStatus(error.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleFeedback(itemId, action) {
    try {
      setFeedbackBusy(true)
      await onFeedbackSaved(itemId, action)
    } finally {
      setFeedbackBusy(false)
    }
  }

  return (
    <section className="panel panel-large">
      <div className="section-header">
        <div>
          <p className="eyebrow">Search</p>
          <h2>Search with natural language</h2>
          <p className="section-copy">Describe the kind of piece or vibe you want, then keep teaching Curate with your likes and dislikes.</p>
        </div>
      </div>
      <div className="search-bar">
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Try phrases like black hoodie, clean streetwear, or smart casual office shirt" />
        <button className="primary-btn" onClick={handleSearch} disabled={loading}>{loading ? 'Searching...' : 'Search'}</button>
      </div>
      <div className="card-grid">
        {items.map((item) => (
          <ItemCard key={item.id} item={item} onFeedback={handleFeedback} busy={feedbackBusy} />
        ))}
      </div>
      {!loading && items.length === 0 ? <p className="muted empty-state">Run a search to see results here.</p> : null}
    </section>
  )
}
