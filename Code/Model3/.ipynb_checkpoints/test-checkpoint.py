import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_exact_same_prices(file_path):
    """
    Analyze how many back-to-back days stocks have EXACTLY the same closing price
    """
    try:
        # Read the data
        print("Loading data...")
        data = pd.read_csv(file_path)
        print(f"Data loaded: {len(data)} rows")
        
        # Get unique symbols
        symbols = data['Symbol'].unique()
        print(f"Found {len(symbols)} unique symbols")
        
        results = []
        
        for i, symbol in enumerate(symbols):
            if i % 50 == 0:
                print(f"Processing symbol {i+1}/{len(symbols)}: {symbol}")
            
            # Get data for this symbol
            stock_data = data[data['Symbol'] == symbol].copy()
            
            if len(stock_data) < 2:
                continue
                
            # Sort by date to ensure proper order
            if 'Date' in stock_data.columns:
                stock_data = stock_data.sort_values('Date')
            
            # Get closing prices
            closes = stock_data['Close'].values
            
            # Find where price stays exactly the same
            same_price_mask = np.diff(closes) == 0
            
            # Count total same-price days
            total_same_days = np.sum(same_price_mask)
            
            # Find consecutive sequences of same prices
            consecutive_sequences = []
            current_sequence_length = 0
            
            for is_same in same_price_mask:
                if is_same:
                    current_sequence_length += 1
                else:
                    if current_sequence_length > 0:
                        consecutive_sequences.append(current_sequence_length)
                    current_sequence_length = 0
            
            # Don't forget the last sequence if it ends with same prices
            if current_sequence_length > 0:
                consecutive_sequences.append(current_sequence_length)
            
            # Calculate statistics
            total_days = len(closes) - 1  # -1 because we're looking at differences
            same_percentage = (total_same_days / total_days * 100) if total_days > 0 else 0
            
            max_consecutive = max(consecutive_sequences) if consecutive_sequences else 0
            num_sequences = len(consecutive_sequences)
            avg_sequence_length = np.mean(consecutive_sequences) if consecutive_sequences else 0
            
            results.append({
                'Symbol': symbol,
                'Total_Days': total_days,
                'Same_Price_Days': total_same_days,
                'Same_Price_Percentage': same_percentage,
                'Num_Sequences': num_sequences,
                'Max_Consecutive_Same': max_consecutive,
                'Avg_Sequence_Length': avg_sequence_length,
                'All_Sequences': consecutive_sequences
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Overall statistics
        print("\n" + "="*60)
        print("EXACT SAME PRICE ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nOverall Statistics:")
        print(f"Total stocks analyzed: {len(results_df)}")
        print(f"Stocks with NO same-price days: {len(results_df[results_df['Same_Price_Days'] == 0])}")
        print(f"Stocks with same-price days: {len(results_df[results_df['Same_Price_Days'] > 0])}")
        
        # Summary stats
        if len(results_df) > 0:
            print(f"\nSame-Price Day Statistics:")
            print(f"Average same-price days per stock: {results_df['Same_Price_Days'].mean():.2f}")
            print(f"Median same-price days per stock: {results_df['Same_Price_Days'].median():.2f}")
            print(f"Max same-price days in one stock: {results_df['Same_Price_Days'].max()}")
            print(f"Average percentage of same-price days: {results_df['Same_Price_Percentage'].mean():.3f}%")
            
            print(f"\nConsecutive Same-Price Sequences:")
            print(f"Average sequences per stock: {results_df['Num_Sequences'].mean():.2f}")
            print(f"Longest consecutive same-price streak: {results_df['Max_Consecutive_Same'].max()} days")
            print(f"Average sequence length: {results_df['Avg_Sequence_Length'].mean():.2f} days")
        
        # Top offenders
        print(f"\nTOP 10 STOCKS WITH MOST SAME-PRICE DAYS:")
        top_same = results_df.nlargest(10, 'Same_Price_Days')[['Symbol', 'Same_Price_Days', 'Same_Price_Percentage', 'Max_Consecutive_Same']]
        print(top_same.to_string(index=False))
        
        print(f"\nTOP 10 STOCKS WITH LONGEST CONSECUTIVE STREAKS:")
        top_streaks = results_df.nlargest(10, 'Max_Consecutive_Same')[['Symbol', 'Max_Consecutive_Same', 'Same_Price_Days', 'Same_Price_Percentage']]
        print(top_streaks.to_string(index=False))
        
        # Distribution analysis
        print(f"\nDISTRIBUTION OF SAME-PRICE DAYS:")
        same_day_counts = Counter(results_df['Same_Price_Days'])
        for days in sorted(same_day_counts.keys())[:20]:  # Show first 20
            count = same_day_counts[days]
            print(f"{days:3d} same-price days: {count:3d} stocks")
        
        if max(same_day_counts.keys()) > 19:
            print("... (showing first 20 categories)")
        
        # Visualization
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Distribution of same-price days
        plt.subplot(2, 2, 1)
        same_days = results_df['Same_Price_Days']
        plt.hist(same_days, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Same-Price Days')
        plt.ylabel('Number of Stocks')
        plt.title('Distribution of Same-Price Days Across Stocks')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Percentage distribution
        plt.subplot(2, 2, 2)
        percentages = results_df['Same_Price_Percentage']
        plt.hist(percentages, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Percentage of Same-Price Days (%)')
        plt.ylabel('Number of Stocks')
        plt.title('Distribution of Same-Price Day Percentages')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Max consecutive streaks
        plt.subplot(2, 2, 3)
        max_streaks = results_df['Max_Consecutive_Same']
        plt.hist(max_streaks, bins=30, alpha=0.7, edgecolor='black', color='green')
        plt.xlabel('Maximum Consecutive Same-Price Days')
        plt.ylabel('Number of Stocks')
        plt.title('Distribution of Longest Same-Price Streaks')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Scatter plot - Total days vs Same-price days
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['Total_Days'], results_df['Same_Price_Days'], alpha=0.6, s=20)
        plt.xlabel('Total Trading Days')
        plt.ylabel('Same-Price Days')
        plt.title('Same-Price Days vs Total Trading Days')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Detailed analysis for extreme cases
        extreme_cases = results_df[results_df['Max_Consecutive_Same'] >= 5]
        if len(extreme_cases) > 0:
            print(f"\n\nSTOCKS WITH 5+ CONSECUTIVE SAME-PRICE DAYS:")
            print("="*60)
            for _, row in extreme_cases.iterrows():
                print(f"{row['Symbol']}: {row['Max_Consecutive_Same']} consecutive days")
                print(f"  All sequences: {row['All_Sequences']}")
                print()
        
        return results_df
        
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        print("Creating sample analysis with dummy data...")
        return create_sample_analysis()
    except Exception as e:
        print(f"Error analyzing data: {e}")
        return None

# Run the analysis
if __name__ == "__main__":
    file_path = '/Users/pedroalexleite/Desktop/Tese/Dados/dataset4.csv'
    analyze_exact_same_prices(file_path)