import { useState, useMemo, useEffect } from 'react';
import './ModelParamsPanel.css';

const tempExamples = [
  {
    low: 'The cat sat on the mat.',
    high: 'A turquoise feline pirouetted atop a velvet rug, pondering quantum dreams.'
  },
  {
    low: 'It is raining today.',
    high: 'Raindrops waltz from the sky, painting the city in shimmering silver.'
  },
  {
    low: 'I like pizza.',
    high: 'Pizza is a cosmic wheel of molten joy and infinite possibility!'
  },
  {
    low: 'The dog barked.',
    high: 'A boisterous hound serenaded the moon with operatic barks.'
  },
  {
    low: 'She opened the door.',
    high: 'With a flourish, she unveiled new worlds behind the ancient door.'
  }
];

function getTempExample(temp) {
  const idx = Math.floor(temp * (tempExamples.length - 1) / 2);
  const ex = tempExamples[idx];
  if (temp < 0.7) return ex.low;
  if (temp > 1.3) return ex.high;
  // Blend
  return `${ex.low.slice(0, Math.floor(ex.low.length/2))}...${ex.high.slice(Math.floor(ex.high.length/2))}`;
}

export default function ModelParamsPanel({ open, onClose, params, setParams, onModelChange }) {
  const [visible, setVisible] = useState(open);
  const [closing, setClosing] = useState(false);
  const [availableModels, setAvailableModels] = useState({});
  const [currentModel, setCurrentModel] = useState('emma');
  const [loadingModel, setLoadingModel] = useState(false);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/models');
        const data = await response.json();
        setAvailableModels(data.available_models || {});
        setCurrentModel(data.current_model || 'emma');
      } catch (error) {
        console.error('Failed to fetch models:', error);
      }
    };
    fetchModels();
  }, []);

  useEffect(() => {
    if (open) {
      setVisible(true);
      setClosing(false);
    } else if (visible) {
      setClosing(true);
      const timeout = setTimeout(() => {
        setVisible(false);
        setClosing(false);
      }, 480); // match animation duration
      return () => clearTimeout(timeout);
    }
  }, [open]);

  const handleModelSwitch = async (modelName) => {
    if (modelName === currentModel) return;
    
    setLoadingModel(true);
    try {
      const response = await fetch('/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentModel(data.current_model);
        if (onModelChange) {
          onModelChange(data.current_model);
        }
      } else {
        console.error('Failed to switch model');
      }
    } catch (error) {
      console.error('Error switching model:', error);
    } finally {
      setLoadingModel(false);
    }
  };

  if (!visible) return null;
  return (
    <div className="paramsOverlay" onClick={onClose}>
      <div className={`paramsPanel modern${closing ? ' closing' : ''}`} onClick={e => e.stopPropagation()}>
        <div className="paramsHeader">
          <span className="paramsTitle">Model Parameters</span>
          <button className="closeBtn" onClick={onClose} aria-label="Close">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M6 6L18 18" stroke="#b0b0b8" strokeWidth="2" strokeLinecap="round"/>
              <path d="M18 6L6 18" stroke="#b0b0b8" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </button>
        </div>
        <div className="paramsDivider" />
        <div className="paramsBody">
          <label className="paramLabel">
            Model
            <div className="modelSelector">
              {Object.entries(availableModels).map(([modelId, modelInfo]) => (
                <div key={modelId} className="modelOptionWrapper">
                  <button
                    className={`modelOption ${currentModel === modelId ? 'active' : ''}`}
                    onClick={() => handleModelSwitch(modelId)}
                    disabled={loadingModel}
                  >
                    <span className="modelOptionText">
                      {modelInfo.name}
                    </span>
                    {currentModel === modelId && (
                      <span className={`activeIndicator ${loadingModel ? 'loading' : ''}`}>
                        {loadingModel ? '⟳' : '✓'}
                      </span>
                    )}
                  </button>
                </div>
              ))}
            </div>
            {loadingModel && (
              <div className="modelLoadingArea" aria-live="polite" aria-busy="true">
                <div className="progressBarContainer">
                  <div className="progressBar">
                    <div className="progressFill"></div>
                  </div>
                </div>
              </div>
            )}
          </label>
          <label className="paramLabel">
            Temperature
            <div className="paramSliderRow">
              <input
                type="range"
                min="0"
                max="2"
                step="0.01"
                value={params.temperature}
                onChange={e => setParams(p => ({ ...p, temperature: Number(e.target.value) }))}
                className="paramSlider"
              />
              <span className="paramValueBubble">{params.temperature}</span>
            </div>
          </label>
          <label className="paramLabel">
            Max Tokens
            <input
              type="number"
              min="1"
              max="512"
              value={params.maxTokens}
              onChange={e => setParams(p => ({ ...p, maxTokens: Number(e.target.value) }))}
              className="paramNumber"
            />
          </label>
        </div>
      </div>
    </div>
  );
}
