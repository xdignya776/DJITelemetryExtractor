import subprocess
import json
import os
import csv
import math
import datetime
from pathlib import Path
import argparse

# Optional imports - will be checked at runtime
GPXPY_AVAILABLE = False
FOLIUM_AVAILABLE = False
OPENCV_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import gpxpy
    import gpxpy.gpx
    GPXPY_AVAILABLE = True
except ImportError:
    pass

try:
    import folium
    from folium.plugins import MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    pass

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


class DJITelemetryExtractor:
    def __init__(self, video_path, output_dir=None, sampling_rate=1):
        """
        Initialize the DJI telemetry extractor
        
        Args:
            video_path: Path to the MP4 video file
            output_dir: Directory to save output files (default: same as video)
            sampling_rate: How often to sample video frames (in seconds)
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = self.video_path.parent
        
        self.sampling_rate = sampling_rate
        self.metadata = None
        self.telemetry = []
        self.video_duration = None
        self.frame_rate = None
    
    def extract_metadata(self):
        """Extract all metadata from the video file using ExifTool"""
        try:
            # Check if ExifTool is available
            subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: ExifTool not found. Please install ExifTool.")
            print("Installation instructions: https://exiftool.org/install.html")
            return False

        try:
            # Run ExifTool and get JSON output with grouped data
            result = subprocess.run(
                ['exiftool', '-j', '-g', self.video_path], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Parse the JSON output
            self.metadata = json.loads(result.stdout)
            if self.metadata:
                self.metadata = self.metadata[0]  # ExifTool returns a list with one item
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting metadata: {e}")
            return False
        except json.JSONDecodeError:
            print("Error parsing ExifTool output")
            return False
    
    def extract_video_info(self):
        """Extract video duration and frame rate"""
        if not self.metadata:
            return False
        
        # Try to get duration
        if 'QuickTime' in self.metadata and 'Duration' in self.metadata['QuickTime']:
            duration_str = self.metadata['QuickTime']['Duration']
            # Convert hh:mm:ss to seconds
            if ':' in duration_str:
                h, m, s = map(float, duration_str.split(':'))
                self.video_duration = h * 3600 + m * 60 + s
            else:
                # Assuming it's already in seconds
                self.video_duration = float(duration_str)
        
        # Try to get frame rate
        if 'QuickTime' in self.metadata and 'VideoFrameRate' in self.metadata['QuickTime']:
            self.frame_rate = float(self.metadata['QuickTime']['VideoFrameRate'])
        
        if self.video_duration and self.frame_rate:
            return True
        
        # If ExifTool didn't work, try FFprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                 '-show_entries', 'stream=duration,r_frame_rate', '-of', 'json', str(self.video_path)],
                capture_output=True, text=True, check=True
            )
            
            ffprobe_data = json.loads(result.stdout)
            if 'streams' in ffprobe_data and ffprobe_data['streams']:
                stream = ffprobe_data['streams'][0]
                
                if 'duration' in stream:
                    self.video_duration = float(stream['duration'])
                
                if 'r_frame_rate' in stream:
                    # r_frame_rate is in the format "num/den"
                    num, den = map(int, stream['r_frame_rate'].split('/'))
                    self.frame_rate = num / den
            
            return bool(self.video_duration and self.frame_rate)
            
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            # Final fallback using OpenCV
            if OPENCV_AVAILABLE:
                try:
                    cap = cv2.VideoCapture(str(self.video_path))
                    self.frame_rate = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.video_duration = frame_count / self.frame_rate
                    cap.release()
                    return bool(self.video_duration and self.frame_rate)
                except Exception as e:
                    print(f"Error extracting video info with OpenCV: {e}")
            
            return False
    
    def extract_dji_telemetry(self):
        """Extract DJI telemetry data from the video metadata"""
        if not self.metadata:
            return False
        
        # Different DJI drones store telemetry in different formats
        # Try XMP metadata first (common in newer DJI drones)
        if 'XMP' in self.metadata:
            xmp = self.metadata['XMP']
            
            # Check for standard GPS data
            if all(key in xmp for key in ['GpsLatitude', 'GpsLongitude']):
                # Simple extraction of single point
                single_point = {
                    'timestamp': xmp.get('CreateDate', ''),
                    'latitude': xmp.get('GpsLatitude', 0),
                    'longitude': xmp.get('GpsLongitude', 0),
                    'altitude': xmp.get('AbsoluteAltitude', 0),
                    'relative_altitude': xmp.get('RelativeAltitude', 0),
                    'gimbal_pitch': xmp.get('GimbalPitchDegree', 0),
                    'gimbal_yaw': xmp.get('GimbalYawDegree', 0),
                    'gimbal_roll': xmp.get('GimbalRollDegree', 0),
                    'flight_yaw': xmp.get('FlightYawDegree', 0),
                    'flight_pitch': xmp.get('FlightPitchDegree', 0),
                    'flight_roll': xmp.get('FlightRollDegree', 0),
                }
                self.telemetry.append(single_point)
            
            # Check for flight logs (common in newer DJI drones)
            if 'FlightXmlLog' in xmp:
                try:
                    # Simple extraction of XML-based flight logs
                    print("Found XML flight log, but parsing not yet implemented")
                    # Advanced parsing would require XML parsing, which varies by drone model
                except Exception as e:
                    print(f"Error parsing XML flight log: {e}")
        
        # Some DJI drones store telemetry in QuickTime metadata
        if 'QuickTime' in self.metadata:
            qt = self.metadata['QuickTime']
            
            # Check for GPS coordinates embedded in location data
            if 'GPSCoordinates' in qt:
                coords = qt['GPSCoordinates']
                if coords and ',' in coords:
                    lat, lon = map(float, coords.split(','))
                    self.telemetry.append({
                        'timestamp': qt.get('CreateDate', ''),
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': qt.get('Altitude', 0),
                    })
        
        # For more advanced extraction, we would need to use specialized libraries
        # like telemetry-parser for DJI, which can extract telemetry at regular intervals
        
        # If no telemetry found, try exiftool's specialized DJI extraction
        if not self.telemetry:
            try:
                # This extraction method works with many DJI drones, extracting SRT format
                # SRT files contain telemetry data at regular intervals
                output_srt = self.output_dir / f"{self.video_path.stem}_telemetry.srt"
                
                with open(output_srt, 'w') as f:
                    subprocess.run(
                        ['exiftool', '-ee', '-p', '${srt:all}', str(self.video_path)],
                        text=True, check=True,
                        stdout=f
                    )
                
                # Parse the SRT file to extract coordinates
                if self.parse_srt_file(output_srt):
                    return True
                else:
                    # If SRT parsing failed, remove the file
                    output_srt.unlink(missing_ok=True)
            except Exception as e:
                print(f"Error extracting DJI telemetry: {e}")
        
        return len(self.telemetry) > 0
    
    def parse_srt_file(self, srt_file):
        """Parse an SRT file to extract GPS coordinates"""
        if not os.path.exists(srt_file):
            return False
        
        try:
            # SRT files have a specific format:
            # 1. Subtitle number
            # 2. Time range (hh:mm:ss,ms --> hh:mm:ss,ms)
            # 3. Content with GPS data
            # 4. Blank line
            
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newline to get each subtitle block
            blocks = content.strip().split('\n\n')
            
            for block in blocks:
                lines = block.strip().split('\n')
                
                if len(lines) < 3:
                    continue
                
                # Parse timestamp
                timestamp_line = lines[1]
                timestamp_parts = timestamp_line.split(' --> ')[0]
                
                # Parse data lines
                data_lines = lines[2:]
                data_text = ' '.join(data_lines)
                
                # Extract GPS data using regex patterns
                # This is a simplified version - actual extraction would depend on the specific
                # format your DJI drone uses in SRT files
                
                # Look for patterns like:
                # GPS(91.12345, 181.54321)
                # latitude:91.12345 longitude:181.54321
                # LAT:91.12345 LNG:181.54321
                
                lat = lon = alt = None
                
                # Try different patterns seen in DJI SRT files
                import re
                
                # Pattern 1: GPS(lat, long)
                gps_match = re.search(r'GPS\s*\(\s*([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)\s*\)', data_text)
                if gps_match:
                    lat, lon = map(float, gps_match.groups())
                
                # Pattern 2: latitude:X longitude:Y
                if not lat or not lon:
                    lat_match = re.search(r'latitude\s*:\s*([+-]?\d+\.\d+)', data_text, re.IGNORECASE)
                    lon_match = re.search(r'longitude\s*:\s*([+-]?\d+\.\d+)', data_text, re.IGNORECASE)
                    
                    if lat_match and lon_match:
                        lat = float(lat_match.group(1))
                        lon = float(lon_match.group(1))
                
                # Pattern 3: LAT:X LNG:Y
                if not lat or not lon:
                    lat_match = re.search(r'LAT\s*:\s*([+-]?\d+\.\d+)', data_text, re.IGNORECASE)
                    lon_match = re.search(r'LNG\s*:\s*([+-]?\d+\.\d+)', data_text, re.IGNORECASE)
                    
                    if lat_match and lon_match:
                        lat = float(lat_match.group(1))
                        lon = float(lon_match.group(1))
                
                # Extract altitude if available
                alt_match = re.search(r'(?:altitude|ALT|height)\s*:\s*([+-]?\d+\.?\d*)', data_text, re.IGNORECASE)
                if alt_match:
                    alt = float(alt_match.group(1))
                
                # Extract additional data
                speed_match = re.search(r'(?:speed|SPD)\s*:\s*([+-]?\d+\.?\d*)', data_text, re.IGNORECASE)
                speed = float(speed_match.group(1)) if speed_match else None
                
                heading_match = re.search(r'(?:heading|HDG|HEADING)\s*:\s*([+-]?\d+\.?\d*)', data_text, re.IGNORECASE)
                heading = float(heading_match.group(1)) if heading_match else None
                
                # Add telemetry point if we found valid coordinates
                if lat is not None and lon is not None:
                    # Parse timestamp to seconds
                    h, m, s_ms = timestamp_parts.split(':')
                    s, ms = s_ms.split(',')
                    timestamp_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                    
                    self.telemetry.append({
                        'timestamp': timestamp_seconds,
                        'latitude': lat,
                        'longitude': lon,
                        'altitude': alt if alt is not None else 0,
                        'speed': speed if speed is not None else 0,
                        'heading': heading if heading is not None else 0,
                    })
            
            return len(self.telemetry) > 0
            
        except Exception as e:
            print(f"Error parsing SRT file: {e}")
            return False
    
    def extract_telemetry_from_video(self):
        """Extract telemetry data from video frames using OpenCV"""
        if not OPENCV_AVAILABLE:
            print("OpenCV not available. Install it for frame-by-frame analysis.")
            return False
        
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            
            # Calculate frame sampling
            frame_sample_interval = int(self.sampling_rate * self.frame_rate)
            
            # Process video frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every Nth frame based on sampling rate
                if frame_count % frame_sample_interval == 0:
                    # This is where you would extract telemetry from the frame
                    # DJI often overlays telemetry data on video frames
                    # Advanced OCR would be needed to extract this information
                    # This is a complex task and beyond the scope of this script
                    timestamp_seconds = frame_count / self.frame_rate
                    
                    # For now, just add the timestamp as a placeholder
                    # Real implementation would extract actual values from the frame
                    pass
                
                frame_count += 1
            
            cap.release()
            return True
            
        except Exception as e:
            print(f"Error extracting telemetry from video: {e}")
            return False
    
    def save_telemetry_to_csv(self):
        """Save extracted telemetry data to a CSV file"""
        if not self.telemetry:
            print("No telemetry data to save")
            return False
        
        try:
            output_csv = self.output_dir / f"{self.video_path.stem}_telemetry.csv"
            
            # Get all unique keys from the telemetry data
            all_keys = set()
            for point in self.telemetry:
                all_keys.update(point.keys())
            
            # Write the data to CSV
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(self.telemetry)
            
            print(f"Telemetry data saved to {output_csv}")
            return True
            
        except Exception as e:
            print(f"Error saving telemetry to CSV: {e}")
            return False
    
    def create_gpx_track(self):
        """Create a GPX track from the telemetry data"""
        if not GPXPY_AVAILABLE:
            print("gpxpy not installed. Install with: pip install gpxpy")
            return False
        
        if not self.telemetry:
            print("No telemetry data to create GPX track")
            return False
        
        try:
            # Create a new GPX object
            gpx = gpxpy.gpx.GPX()
            
            # Create a track
            track = gpxpy.gpx.GPXTrack()
            gpx.tracks.append(track)
            
            # Create a segment in the track
            segment = gpxpy.gpx.GPXTrackSegment()
            track.segments.append(segment)
            
            # Create points in the segment
            for point in self.telemetry:
                # Skip invalid points
                if 'latitude' not in point or 'longitude' not in point:
                    continue
                
                lat = point['latitude']
                lon = point['longitude']
                
                # Skip points with invalid coordinates
                if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    continue
                
                # Create a point
                gpx_point = gpxpy.gpx.GPXTrackPoint(
                    latitude=lat,
                    longitude=lon,
                    elevation=point.get('altitude', 0)
                )
                
                # Add the point to the segment
                segment.points.append(gpx_point)
            
            # Save the GPX track
            output_gpx = self.output_dir / f"{self.video_path.stem}_track.gpx"
            with open(output_gpx, 'w') as f:
                f.write(gpx.to_xml())
            
            print(f"GPX track saved to {output_gpx}")
            return True
            
        except Exception as e:
            print(f"Error creating GPX track: {e}")
            return False
    
    def create_map_visualization(self):
        """Create an interactive map visualization of the flight path"""
        if not FOLIUM_AVAILABLE:
            print("Folium not installed. Install with: pip install folium")
            return False
        
        if not self.telemetry:
            print("No telemetry data to create map visualization")
            return False
        
        try:
            # Extract coordinates for map
            coords = []
            for point in self.telemetry:
                if 'latitude' in point and 'longitude' in point:
                    lat = point['latitude']
                    lon = point['longitude']
                    
                    # Skip invalid coordinates
                    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                        continue
                    
                    coords.append((lat, lon))
            
            if not coords:
                print("No valid coordinates found in telemetry data")
                return False
            
            # Calculate the center of the map
            center_lat = sum(lat for lat, _ in coords) / len(coords)
            center_lon = sum(lon for _, lon in coords) / len(coords)
            
            # Create a map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
            
            # Add flight path
            folium.PolyLine(
                coords,
                weight=3,
                color='red',
                opacity=0.7,
                tooltip='Flight Path'
            ).add_to(m)
            
            # Add markers for start and end points
            folium.Marker(
                coords[0],
                popup='Start',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            folium.Marker(
                coords[-1],
                popup='End',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
            
            # Save the map
            output_html = self.output_dir / f"{self.video_path.stem}_map.html"
            m.save(str(output_html))
            
            print(f"Map visualization saved to {output_html}")
            return True
            
        except Exception as e:
            print(f"Error creating map visualization: {e}")
            return False
    
    def create_altitude_plot(self):
        """Create a plot of altitude over time"""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not installed. Install with: pip install matplotlib")
            return False
        
        if not self.telemetry:
            print("No telemetry data to create altitude plot")
            return False
        
        try:
            # Extract altitude data
            timestamps = []
            altitudes = []
            
            for point in self.telemetry:
                if 'timestamp' in point and 'altitude' in point:
                    timestamps.append(point['timestamp'])
                    altitudes.append(point['altitude'])
            
            if not timestamps or not altitudes:
                print("No altitude data found in telemetry")
                return False
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, altitudes, '-b')
            plt.title('Drone Altitude Over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Altitude (meters)')
            plt.grid(True)
            
            # Save the plot
            output_plot = self.output_dir / f"{self.video_path.stem}_altitude.png"
            plt.savefig(output_plot)
            plt.close()
            
            print(f"Altitude plot saved to {output_plot}")
            return True
            
        except Exception as e:
            print(f"Error creating altitude plot: {e}")
            return False
    
    def process(self):
        """Process the video to extract and visualize telemetry data"""
        print(f"Processing video: {self.video_path}")
        
        # Step 1: Extract metadata
        if not self.extract_metadata():
            print("Failed to extract metadata")
            return False
        
        # Step 2: Extract video info
        if not self.extract_video_info():
            print("Failed to extract video info")
            # Continue anyway as it's not critical
        
        # Step 3: Extract DJI telemetry
        if not self.extract_dji_telemetry():
            print("Failed to extract DJI telemetry")
            return False
        
        # Step 4: Save telemetry to CSV
        self.save_telemetry_to_csv()
        
        # Step 5: Create GPX track
        if GPXPY_AVAILABLE:
            self.create_gpx_track()
        else:
            print("gpxpy not installed. Skipping GPX track creation.")
        
        # Step 6: Create map visualization
        if FOLIUM_AVAILABLE:
            self.create_map_visualization()
        else:
            print("folium not installed. Skipping map visualization.")
        
        # Step 7: Create altitude plot
        if MATPLOTLIB_AVAILABLE:
            self.create_altitude_plot()
        else:
            print("matplotlib not installed. Skipping altitude plot.")
        
        print(f"Processing complete for {self.video_path}")
        return True


def check_dependencies():
    """Check if the required dependencies are installed"""
    missing_deps = []
    
    # Check for ExifTool
    try:
        subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_deps.append("ExifTool (https://exiftool.org/install.html)")
    
    # Check for Python packages
    if not GPXPY_AVAILABLE:
        missing_deps.append("gpxpy (pip install gpxpy)")
    
    if not FOLIUM_AVAILABLE:
        missing_deps.append("folium (pip install folium)")
    
    if not OPENCV_AVAILABLE:
        missing_deps.append("OpenCV (pip install opencv-python)")
    
    if not MATPLOTLIB_AVAILABLE:
        missing_deps.append("matplotlib (pip install matplotlib)")
    
    return missing_deps


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Extract and visualize telemetry data from DJI drone videos')
    parser.add_argument('video', type=str, help='Path to the DJI drone video file')
    parser.add_argument('-o', '--output', type=str, help='Output directory for extracted data')
    parser.add_argument('-s', '--sampling', type=float, default=1.0, help='Sampling rate in seconds (default: 1.0)')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    # Check for dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("Missing dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        if args.install_deps:
            print("\nInstalling Python dependencies...")
            subprocess.call(['pip', 'install', 'gpxpy', 'folium', 'opencv-python', 'matplotlib'])
            print("\nNote: ExifTool must be installed manually if missing.")
            print("Visit https://exiftool.org/install.html for installation instructions.")
        else:
            print("\nInstall dependencies manually or run with --install-deps")
            return
    
    # Process the video
    extractor = DJITelemetryExtractor(
        video_path=args.video,
        output_dir=args.output,
        sampling_rate=args.sampling
    )
    
    extractor.process()


if __name__ == "__main__":
    main()