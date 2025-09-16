#!/usr/bin/env python3
"""
Test script for PerceiverEncoder latent initialization methods.
This script validates that all initialization methods work correctly.
"""

import torch
from src.deepmuonreco.nn.transformers.perceiver import PerceiverEncoder
from src.deepmuonreco.nn.models.latent_attention import LatentAttentionModel


def test_perceiver_encoder_initialization():
    """Test PerceiverEncoder with different initialization methods."""
    print("Testing PerceiverEncoder initialization methods...")
    
    init_methods = [
        'normal',
        'xavier_uniform', 
        'xavier_normal',
        'kaiming_uniform',
        'kaiming_normal',
        'truncated_normal',
        'zeros'
    ]
    
    expected_ranges = {
        'normal': (0.8, 1.2),  # Should be close to std=1
        'xavier_uniform': (0.1, 0.3),  # Smaller range
        'xavier_normal': (0.1, 0.3),   # Similar to xavier_uniform
        'kaiming_uniform': (0.1, 0.3),  # Similar range
        'kaiming_normal': (0.1, 0.3),   # Similar range
        'truncated_normal': (0.01, 0.03),  # Very small range
        'zeros': (0.0, 0.0),  # Exactly zero
    }
    
    for method in init_methods:
        encoder = PerceiverEncoder(
            latent_len=16,
            latent_dim=64,
            num_heads=4,
            latent_init=method
        )
        
        std = encoder.latent.std().item()
        min_val = encoder.latent.min().item()
        max_val = encoder.latent.max().item()
        mean_val = encoder.latent.mean().item()
        
        print(f"  {method}: std={std:.4f}, mean={mean_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
        
        # Validate expected range
        expected_min, expected_max = expected_ranges[method]
        assert expected_min <= std <= expected_max, f"Std {std} not in expected range {expected_ranges[method]} for {method}"
        
        # Test forward pass
        batch_size = 2
        seq_len = 10
        input_dim = 64  # Should match latent_dim
        
        input_tensor = torch.randn(batch_size, seq_len, input_dim)
        data_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        with torch.no_grad():
            output = encoder(input_tensor, data_mask)
            assert output.shape == (batch_size, 16, 64), f"Unexpected output shape: {output.shape}"
    
    print("  ✓ All PerceiverEncoder initialization methods passed!")


def test_invalid_initialization():
    """Test that invalid initialization methods raise ValueError."""
    print("Testing invalid initialization method...")
    
    try:
        encoder = PerceiverEncoder(
            latent_len=10,
            latent_dim=32,
            num_heads=2,
            latent_init='invalid_method'
        )
        assert False, "Should have raised ValueError for invalid method"
    except ValueError as e:
        assert "Unknown latent initialization method" in str(e)
        print("  ✓ Invalid method properly rejected!")


def test_latent_attention_model_initialization():
    """Test LatentAttentionModel with different initialization methods."""
    print("Testing LatentAttentionModel initialization...")
    
    init_methods = ['normal', 'xavier_uniform', 'truncated_normal']
    
    for method in init_methods:
        model = LatentAttentionModel(
            track_dim=3,
            segment_dim=6,
            hit_dim=3,
            output_dim=1,
            model_dim=32,
            num_heads=2,
            track_latent_len=8,
            muon_det_latent_len=4,
            encoder_num_layers=1,
            decoder_num_layers=1,
            latent_init=method
        )
        
        # Test forward pass
        batch_size = 2
        track = torch.randn(batch_size, 5, 3)
        track_mask = torch.ones(batch_size, 5, dtype=torch.bool)
        segment = torch.randn(batch_size, 4, 6)
        segment_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        rechit = torch.randn(batch_size, 6, 3)
        rechit_mask = torch.ones(batch_size, 6, dtype=torch.bool)
        
        with torch.no_grad():
            output = model(track, track_mask, segment, segment_mask, rechit, rechit_mask)
            assert output.shape == (batch_size, 5), f"Unexpected output shape: {output.shape}"
        
        print(f"  ✓ {method} initialization works with LatentAttentionModel")
    
    print("  ✓ All LatentAttentionModel tests passed!")


if __name__ == "__main__":
    test_perceiver_encoder_initialization()
    test_invalid_initialization()
    test_latent_attention_model_initialization()
    print("\n🎉 All tests passed successfully!")