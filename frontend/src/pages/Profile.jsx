import { useEffect, useMemo, useState } from 'react'
import { fetchOptions, saveProfile, uploadStyleImage, fetchStyleProfile, toUploadedStyleImageUrl, fetchPreferences } from '../api'

const EMPTY_PROFILE = {
  gender: 'All',
  preferred_colors: [],
  disliked_colors: [],
  preferred_categories: [],
  preferred_types: [],
  preferred_usage: [],
}

function MultiSelect({ label, options, value, onChange }) {
  const normalized = value || []
  return (
    <label className="field-block">
      <span>{label}</span>
      <select multiple value={normalized} onChange={(e) => onChange(Array.from(e.target.selectedOptions, opt => opt.value))}>
        {options?.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  )
}

export default function Profile({ userId, onStatus, onProfileSaved }) {
  const [options, setOptions] = useState({ genders: [], colors: [], categories: [], types: [], usage: [] })
  const [profile, setProfile] = useState(EMPTY_PROFILE)
  const [saving, setSaving] = useState(false)
  const [styleLearning, setStyleLearning] = useState({
    learned_colors: [],
    brightness_label: '',
    contrast_label: '',
    vibe_label: '',
    image_url: '',
  })

  useEffect(() => {
    fetchOptions()
      .then((data) => setOptions(data))
      .catch((error) => onStatus(error.message))
  }, [onStatus])

  useEffect(() => {
    fetchPreferences(userId)
      .then((data) => setProfile(data))
      .catch(() => {})
  }, [userId])

  useEffect(() => {
    fetchStyleProfile(userId)
      .then((data) => setStyleLearning(data))
      .catch(() => {})
  }, [userId])

  const genderOptions = useMemo(() => options.genders || ['All'], [options.genders])

  async function handleSave() {
    setSaving(true)
    try {
      await saveProfile(userId, profile)
      onStatus('Profile saved.')
      onProfileSaved?.()
    } catch (error) {
      onStatus(error.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleImageUpload(event) {
    const file = event.target.files?.[0]
    if (!file) return
    try {
      const result = await uploadStyleImage(userId, file)
      setStyleLearning(result)
      onStatus('Style image uploaded and analyzed.')
      onProfileSaved?.()
    } catch (error) {
      onStatus(error.message)
    }
  }

  return (
    <section className="panel panel-large">
      <div className="section-header">
        <div>
          <p className="eyebrow">Profile</p>
          <h2>Build your style profile</h2>
          <p className="section-copy">Save your taste once, come back with the same user ID, and Curate will pick up where you left off.</p>
        </div>
        <button className="primary-btn" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : 'Save profile'}
        </button>
      </div>

      <div className="saved-box">
        <h3>Saved preferences</h3>
        <div className="pill-row">
          {(profile.preferred_colors || []).map((x) => <span className="pill" key={`color-${x}`}>{x}</span>)}
          {(profile.disliked_colors || []).map((x) => <span className="pill" key={`avoid-${x}`}>Avoid: {x}</span>)}
          {(profile.preferred_categories || []).map((x) => <span className="pill" key={`cat-${x}`}>{x}</span>)}
          {(profile.preferred_types || []).map((x) => <span className="pill" key={`type-${x}`}>{x}</span>)}
          {(profile.preferred_usage || []).map((x) => <span className="pill" key={`usage-${x}`}>{x}</span>)}
          {profile.gender && profile.gender !== 'All' ? <span className="pill">Fit: {profile.gender}</span> : null}
          {!profile.preferred_colors?.length &&
          !profile.disliked_colors?.length &&
          !profile.preferred_categories?.length &&
          !profile.preferred_types?.length &&
          !profile.preferred_usage?.length &&
          (!profile.gender || profile.gender === 'All') ? (
            <span className="muted">No saved preferences yet.</span>
          ) : null}
        </div>
      </div>

      <div className="profile-grid">
        <label className="field-block">
          <span>Gender / fit preference</span>
          <select value={profile.gender || 'All'} onChange={(e) => setProfile({ ...profile, gender: e.target.value })}>
            {genderOptions.map((option) => <option key={option} value={option}>{option}</option>)}
          </select>
        </label>

        <MultiSelect label="Preferred colors" options={options.colors || []} value={profile.preferred_colors} onChange={(vals) => setProfile({ ...profile, preferred_colors: vals })} />
        <MultiSelect label="Colors to avoid" options={options.colors || []} value={profile.disliked_colors} onChange={(vals) => setProfile({ ...profile, disliked_colors: vals })} />
        <MultiSelect label="Preferred categories" options={options.categories || []} value={profile.preferred_categories} onChange={(vals) => setProfile({ ...profile, preferred_categories: vals })} />
        <MultiSelect label="Preferred item types" options={options.types || []} value={profile.preferred_types} onChange={(vals) => setProfile({ ...profile, preferred_types: vals })} />
        <MultiSelect label="Preferred usage / occasion" options={options.usage || []} value={profile.preferred_usage} onChange={(vals) => setProfile({ ...profile, preferred_usage: vals })} />
      </div>

      <div className="upload-panel">
        <div>
          <h3>Upload style inspiration</h3>
          <p className="muted">Upload an image and the system will learn a few visual signals from it.</p>
        </div>
        <input type="file" accept="image/*" onChange={handleImageUpload} />
      </div>

      <div className="panel inner-panel">
        <h3>What the algorithm is learning from your image</h3>
        <p className="muted">This visual summary is used as an extra preference signal for recommendations.</p>

        {styleLearning.image_url ? (
          <div className="preview-wrap">
            <img
              src={toUploadedStyleImageUrl(styleLearning.image_url)}
              alt="Uploaded style reference"
              className="uploaded-preview"
            />
          </div>
        ) : null}

        <div className="pill-row top-gap">
          {(styleLearning.learned_colors || []).map((color) => (
            <span className="pill" key={color}>{color}</span>
          ))}
          {styleLearning.brightness_label ? <span className="pill">{styleLearning.brightness_label}</span> : null}
          {styleLearning.contrast_label ? <span className="pill">{styleLearning.contrast_label}</span> : null}
          {styleLearning.vibe_label ? <span className="pill">{styleLearning.vibe_label}</span> : null}
        </div>
      </div>
    </section>
  )
}
