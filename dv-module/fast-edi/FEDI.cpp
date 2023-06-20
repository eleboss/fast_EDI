#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/core.hpp>

#include <iostream>
#include <fstream>
#include <math.h> 
#include <ctime>

#define DAVIS346_IMAGE_HEIGHT 260
#define DAVIS346_IMAGE_WIDTH 346
#define DAVIS346_IMAGE_PIXNUM (DAVIS346_IMAGE_HEIGHT * DAVIS346_IMAGE_WIDTH)
#define TIME_SCALE 1e6
#define MAX_SIZE 700
typedef std::numeric_limits< double > dbl;

class FastEDI : public dv::ModuleBase {

private:
	bool frameReceiveFlag_ = false;
	bool pushbackFlag_ = false;
	double frame_ts_expStart = INT64_MAX;
	double frame_ts_expEnd = INT64_MAX;
	double frame_ts = 0;

	cv::Mat input_img;
	long int nonuni_counter = 0;

	long long total_events_decode = 0;
	long long total_runtime_decode = 0;


	std::vector<double> crf_exposure;
	std::vector<double> crf_irr;
	std::vector<double> aps_irr_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	// std::vector<uint8_t> aps_img = std::vector<uint8_t>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<uint8_t> deblur_exp = std::vector<uint8_t>(DAVIS346_IMAGE_PIXNUM, 1);

	// int dn_upthresh_;
	// int dn_downthresh_;
	double dvs_c_pos_;
	double dvs_c_neg_;
	bool loadOnceFlag_ = true;
	std::string crf_addr;

	// variable for event runtime processing
	std::vector<double> ev_irr_runtime_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> stacked_ev_irr_runtime_exp = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);
	double stacked_runtime_counter_nonuni = 0;


	std::vector<double> dynamic_runtime_counter = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);

	std::array<std::array<double, 2>, MAX_SIZE * DAVIS346_IMAGE_PIXNUM> stacked_runtime_counter_nonuni_list;

	std::array<std::array<std::array<double,3>,MAX_SIZE>,DAVIS346_IMAGE_PIXNUM> ev_irr_exp_runtime_list;



	// variable for irrdiance estimation
	std::vector<double>  est_ev_irr_exp_nonuni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> est_ev_pixcount_uni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);

	std::vector<long> runtime_list;


	double est_ev_pixcount_nonuni = 0;

	// variable for irrdiance estimation, aps side
	std::vector<double> aps_irr_deblur_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);


public:
	FastEDI() {
		outputs.getFrameOutput("image").setup(DAVIS346_IMAGE_WIDTH, DAVIS346_IMAGE_HEIGHT, "deblur result for vis, x*y");

	}
	std::vector<double> loading_vec_from_file(std::string file_path)
	{
		char sep = ' ';
		std::string line;
		std::vector<double> vec;
		double vec_value;
		std::ifstream myfile (file_path);
		int count;
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				count = 0;
				for(size_t p=0, q=0; p!=line.npos; p=q)
				{
					std::string::size_type sz;     // alias of size_t
					if(count == 0)
					{
					vec_value = stold(line.substr(p+(p!=0), (q=line.find(sep, p+1))-p-(p!=0)),&sz);
					}
					count += 1; 
				}
				vec.push_back(vec_value);
			}
			myfile.close();
		}
		return vec;
	}
	static void initInputs(dv::InputDefinitionList &in) {
		in.addFrameInput("frames");
		in.addEventInput("events");
	}
	static void initOutputs(dv::OutputDefinitionList &out) {
		out.addFrameOutput("image");
	}

	static const char *initDescription() {
		return ("The fast EDI pipeline");
	}

	static void initConfigOptions(dv::RuntimeConfig &config) {
		config.add("dvs_c_pos",
			dv::ConfigOption::doubleOption(
				"DVS contrast positive.", 0.2609, 0, 10));
		config.add("dvs_c_neg",
			dv::ConfigOption::doubleOption(
				"DVS contrast negative.", -0.2415, -10, 0));
		config.add("crf_address", 
			dv::ConfigOption::stringOption(
				"location of the camera response function", "/home/eleboss/Documents/fast_EDI/camera_response_function/crf.txt"));

	}
	void configUpdate() override {
		// get parameter
		dvs_c_pos_ = config.getDouble("dvs_c_pos");
		dvs_c_neg_ = config.getDouble("dvs_c_neg");
		crf_addr = config.getString("crf_address");

		if(loadOnceFlag_)
		{
			loadOnceFlag_ = false;
			crf_exposure = loading_vec_from_file(crf_addr);
		}

	}

	void run() override 
	{
		double exposure_time = 0;
		// get data
		const auto frame = inputs.getFrameInput("frames").frame();

		if(frame && !frameReceiveFlag_)
		{
			
			input_img = *frame.getMatPointer();

			frame_ts = double(frame.timestamp())/TIME_SCALE;
			frame_ts_expStart = double(frame.timestampStartOfExposure())/TIME_SCALE;
			frame_ts_expEnd = double(frame.timestampEndOfExposure())/TIME_SCALE;

			exposure_time = frame_ts_expEnd - frame_ts_expStart;

			crf_irr = crf_exposure;
			double log_exposure_time = std::log(exposure_time);
			transform(crf_irr.begin(), crf_irr.end(), crf_irr.begin(), bind2nd(std::minus<double>(), log_exposure_time)); 

			std::fill(aps_irr_log.begin(), aps_irr_log.end(), 0);
			// crf mapping
			for(int j = 0; j < DAVIS346_IMAGE_PIXNUM; j++) 
			{
				uint8_t pix_dn = input_img.at<uint8_t>(j);
				// log.info<<pix_dn<<" "<< dv::logEnd;
				aps_irr_log[j] = crf_irr[pix_dn];
				// aps_img[j] = pix_dn;
			}
			frameReceiveFlag_ = true;
		}

		auto events = inputs.getEventInput("events").events();
		if(events)
		{
			double latest_event_time = double(events[-1].timestamp())/TIME_SCALE;
			double first_event_time = double(events[0].timestamp())/TIME_SCALE;
			///////////////////////////
			// int ev_counter = 0;
			// auto start = std::chrono::high_resolution_clock::now();
			for (const dv::Event &event : events)
			{

				int ev_x_rt = event.x();
				int ev_y_rt = event.y();
				double ev_t_rt = double(event.timestamp())/TIME_SCALE;
				int pix_index = ev_y_rt * DAVIS346_IMAGE_WIDTH + ev_x_rt;
				ev_irr_runtime_log[pix_index] = ev_irr_runtime_log[pix_index] + (event.polarity() == true ? dvs_c_pos_ : dvs_c_neg_);
				double ev_irr_exp_rt = exp(ev_irr_runtime_log[pix_index]);
				stacked_ev_irr_runtime_exp[pix_index] = stacked_ev_irr_runtime_exp[pix_index] + ev_irr_exp_rt;
				stacked_runtime_counter_nonuni = stacked_runtime_counter_nonuni + 1;

				int runtime_counter = dynamic_runtime_counter[pix_index];
				ev_irr_exp_runtime_list[pix_index][runtime_counter] = {ev_t_rt, ev_irr_exp_rt, stacked_runtime_counter_nonuni};
				stacked_runtime_counter_nonuni_list[nonuni_counter] = {ev_t_rt, stacked_runtime_counter_nonuni};
				dynamic_runtime_counter[pix_index] += 1;
				nonuni_counter += 1;
			}

			if(latest_event_time > frame_ts_expEnd && frameReceiveFlag_)
			{

				auto start = std::chrono::high_resolution_clock::now();

				frameReceiveFlag_ = false;
				pushbackFlag_ = false;
				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// locate nonuni init
				double stacked_counter_nonuni_mark_in = 0;
				double stacked_counter_nonuni_mark_out = 1; // no event, no change, no count, set to 0 for good debug
				for(int i = nonuni_counter-1; i > 1; i--)
				{
					if(stacked_runtime_counter_nonuni_list[i][0] <= frame_ts_expStart)
					{
						stacked_counter_nonuni_mark_in = stacked_runtime_counter_nonuni_list[i][1];
						break;
					}
					if(stacked_runtime_counter_nonuni_list[i][0] <= frame_ts_expEnd && (stacked_counter_nonuni_mark_out==1))
					{
						stacked_counter_nonuni_mark_out = stacked_runtime_counter_nonuni_list[i][1];
					}
				}
				est_ev_pixcount_nonuni = stacked_counter_nonuni_mark_out - stacked_counter_nonuni_mark_in;
				///////////////////
				for(int i = 0; i < DAVIS346_IMAGE_PIXNUM; i++) 
				{
					bool minTsFlag_ = false;
					bool maxTsFlag_ = false;
					int ev_ts_in = 0;
					int ev_ts_out = 0;
					int ev_ts_before_in = 0;
					// int ev_ts_after_out = 0;
					double est_ev_irr_exp_in = 0;
					double est_ev_pixcount_uni_in = 0;
					double est_ev_pixcount_uni_out = 0;
					double est_ev_irr_exp_out = 0;
					for(int ev = 1; ev < dynamic_runtime_counter[i]; ev++)
					{
						const double ev_t = ev_irr_exp_runtime_list[i][ev][0];
						if(!minTsFlag_ && ev_t >= frame_ts_expStart && ev_t < frame_ts_expEnd)
						{
							minTsFlag_ = true;
							ev_ts_in = ev;
							ev_ts_out = ev;
						}
						if(!maxTsFlag_ && minTsFlag_ && ev_t == frame_ts_expEnd)
						{
							maxTsFlag_ = true;
							ev_ts_out = ev;
						}
						else if (!maxTsFlag_ && minTsFlag_ && ev_t > frame_ts_expEnd)
						{
							maxTsFlag_ = true;
							ev_ts_out = ev-1;

						}
						if(minTsFlag_ && !maxTsFlag_ && ev == dynamic_runtime_counter[i]-1)
						{
							ev_ts_out = ev;
						}

						if(ev_t < frame_ts_expStart)
						{
							ev_ts_before_in = ev;
						}
					}

					if(ev_ts_in > 1)
					{
						double event_irr_exp_init_mark = ev_irr_exp_runtime_list[i][ev_ts_in-1][1];
						for(int j = ev_ts_in; j < ev_ts_out; j++)
						{
							// (j+1 - j) * j
							double event_irr_exp_mid = ev_irr_exp_runtime_list[i][j][1] / event_irr_exp_init_mark;
							double nui_c_front = ev_irr_exp_runtime_list[i][j][2] - stacked_counter_nonuni_mark_in;
							double nui_c_end = ev_irr_exp_runtime_list[i][j+1][2] - stacked_counter_nonuni_mark_in;
							est_ev_irr_exp_nonuni[i] = est_ev_irr_exp_nonuni[i] + (nui_c_end - nui_c_front) * event_irr_exp_mid;
						}
						double event_irr_exp_end = ev_irr_exp_runtime_list[i][ev_ts_out][1] / event_irr_exp_init_mark;
						double event_counter_nonuni_end = ev_irr_exp_runtime_list[i][ev_ts_out][2] - stacked_counter_nonuni_mark_in;
						double event_counter_nonuni_start = ev_irr_exp_runtime_list[i][ev_ts_in][2] - stacked_counter_nonuni_mark_in;
						est_ev_irr_exp_nonuni[i] = est_ev_irr_exp_nonuni[i] 
													+ (est_ev_pixcount_nonuni - event_counter_nonuni_end) * event_irr_exp_end
													+  event_counter_nonuni_start;
					}
					else if(ev_ts_in == 1)
					{
						for(int j = ev_ts_in; j < ev_ts_out; j++)
						{
							// (j+1 - j) * j
							double event_irr_exp_mid = ev_irr_exp_runtime_list[i][j][1];
							double nui_c_front = ev_irr_exp_runtime_list[i][j][2] - stacked_counter_nonuni_mark_in;
							double nui_c_end = ev_irr_exp_runtime_list[i][j+1][2] - stacked_counter_nonuni_mark_in;
							est_ev_irr_exp_nonuni[i] = est_ev_irr_exp_nonuni[i]
												+ (nui_c_end - nui_c_front) * event_irr_exp_mid;
						}
						double event_irr_exp_end = ev_irr_exp_runtime_list[i][ev_ts_out][1];
						double event_counter_nonuni_end = ev_irr_exp_runtime_list[i][ev_ts_out][2] - stacked_counter_nonuni_mark_in;
						double event_counter_nonuni_start = ev_irr_exp_runtime_list[i][ev_ts_in][2] - stacked_counter_nonuni_mark_in;
						est_ev_irr_exp_nonuni[i] = est_ev_irr_exp_nonuni[i]
													+ (est_ev_pixcount_nonuni - event_counter_nonuni_end) * event_irr_exp_end
													+  event_counter_nonuni_start;
					}
					else if(ev_ts_in == 0)
					{
						est_ev_irr_exp_nonuni[i] = est_ev_pixcount_nonuni;
					}
					// the hot pixel will set irr_exp too small, make it meansless, thus just set it to 1.
					if(std::isnan(est_ev_irr_exp_nonuni[i]))
						est_ev_irr_exp_nonuni[i] = est_ev_pixcount_nonuni;
					// --------------------------- nonuni nor
					est_ev_irr_exp_nonuni[i] = est_ev_irr_exp_nonuni[i] / est_ev_pixcount_nonuni;
					// --------------------------- 
					//////////////////////////////////// pixcheck and deblur
					aps_irr_deblur_log[i] = aps_irr_log[i] - std::log(est_ev_irr_exp_nonuni[i]);
				}
				
				// auto elapsed = std::chrono::high_resolution_clock::now() - start;
				// long long runtime_duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				// total_events_decode += nonuni_counter;
				// total_runtime_decode += runtime_duration;
				// double event_rate_decode = total_events_decode / (double(total_runtime_decode)/TIME_SCALE);
				// log.info << "total_runtime_decode: " << total_runtime_decode<< "event rate decode: "<< event_rate_decode << dv::logEnd;

				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// clear parts of the container 
				stacked_runtime_counter_nonuni = 0;
				std::fill(ev_irr_runtime_log.begin(), ev_irr_runtime_log.end(), 0);
				std::fill(stacked_ev_irr_runtime_exp.begin(), stacked_ev_irr_runtime_exp.end(), 1);
				nonuni_counter = 0;
				dynamic_runtime_counter = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
		
				////////////////////////////////////////// output as image
				for (int i = 0; i < DAVIS346_IMAGE_PIXNUM; i++)
				{
					for (uint8_t j = 0; j < 255; j++)
					{
						if(aps_irr_deblur_log[i] >= crf_irr[j])
							deblur_exp[i] = j;
						else
							break;
					}
				}
				//load to mat
				cv::Mat deblur_exp_mat(1, deblur_exp.size(), CV_8UC1, deblur_exp.data());
				deblur_exp_mat = deblur_exp_mat.reshape(1,DAVIS346_IMAGE_HEIGHT);

				// send it for exposure estimation
				auto outFrame_deblur = outputs.getFrameOutput("image").frame();
				outFrame_deblur.setTimestamp(frame_ts * TIME_SCALE);
				outFrame_deblur.setMat(deblur_exp_mat);
				outFrame_deblur.commit();
			}
		}
	}
};

registerModuleClass(FastEDI)
