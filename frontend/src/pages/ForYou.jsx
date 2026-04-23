import { useEffect, useState } from 'react'
import { recommend } from '../api'
import ItemCard from '../components/ItemCard'

export default function ForYou({ userId, refreshKey, onStatus, onFeedbackSaved }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [feedbackBusy, setFeedbackBusy] = useState(false)

  useEffect(() => {
    async function loadRecommendations() {
      setLoading(true)
      try {
        const data = await recommend(userId)
        setItems(Array.isArray(data) ? data : [])
      } catch (error) {
        onStatus(error.message)
      } finally {
        setLoading(false)
      }
    }
    loadRecommendations()
  }, [userId, refreshKey, onStatus])

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
          <p className="eyebrow">For You</p>
          <h2>Your personalized feed</h2>
          <p className="section-copy">This section becomes more accurate as Curate learns from your saved profile, uploaded inspiration, and feedback history.</p>
        </div>
      </div>
      {loading ? <p className="muted">Loading recommendations...</p> : null}
      {!loading && items.length === 0 ? (
        <p className="muted empty-state">Add profile preferences or rate a few items first to unlock personalized recommendations.</p>
      ) : null}
      <div className="card-grid">
        {items.map((item) => (
          <ItemCard key={item.id} item={item} onFeedback={handleFeedback} busy={feedbackBusy} />
        ))}
      </div>
    </section>
  )
}
