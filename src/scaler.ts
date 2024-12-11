export class MinMaxScaler {
    private min: number;
    private max: number;
  
    constructor() {
      this.min = 0;
      this.max = 0;
    }
  
    // Fit the scaler to the data
    fit(data: number[][]): void {
      const mins = data.map((row) => Math.min(...row));
      const maxs = data.map((row) => Math.max(...row));
  
      this.min = Math.min(...mins);
      this.max = Math.max(...maxs);
    }
  
    // Transform the data using min-max scaling
    transform(data: number[][]): number[][] {
      return data.map((row) => row.map((value) => (value - this.min) / (this.max - this.min)));
    }
  
    // Inverse transform the normalized data
    inverseTransform(data: number[][]): number[][] {
      return data.map((row) => row.map((value) => value * (this.max - this.min) + this.min));
    }
  }
  