#include <dv-sdk/module.hpp>
#include <dv-sdk/processing/core.hpp>

#include <iostream>
#include <fstream>
#include <math.h> 
#include <ctime>

#include <libcaercpp/devices/davis.hpp>
#include <libcaercpp/devices/device_discover.hpp>

#include "dvec_base.hpp"

#define DAVIS346_IMAGE_HEIGHT 260
#define DAVIS346_IMAGE_WIDTH 346
#define DAVIS346_IMAGE_PIXNUM (DAVIS346_IMAGE_HEIGHT * DAVIS346_IMAGE_WIDTH)
#define TIME_SCALE 1e6
#define MAX_SIZE 700
typedef std::numeric_limits< double > dbl;

class SceneIrrdianceEst : public dv::ModuleBase {

private:
	long numberOfEvents = 0;
	bool frameReceiveFlag_ = false;
	bool enoughEventFlag_ = false;
	bool pushbackFlag_ = false;
	bool frontEndSwitch = true; // front is true
	double trig_aps_start_ts = 0;
	double trig_aps_end_ts = 0;
	bool push_or_trigger = 0; // 0 for trigger
	dv::EventStore store;
	double frame_ts_expStart = INT64_MAX;
	double frame_ts_expEnd = INT64_MAX;
	double frame_ts_frStart = INT64_MAX;
	double frame_ts_frEnd = INT64_MAX;
	double frame_ts_frEnd_prev = 0;
	double frame_ts = 0;
	int freq_counter = 0;
	std::vector<double> frame_ts_expStart_list;
	std::vector<double> frame_ts_expEnd_list;
	std::vector<double>	frame_ts_frStart_list;
	std::vector<double> frame_ts_frEnd_list;

	std::vector<double> temp_event_img_nonuni =  std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> temp_event_img_exp_nonuni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);
	std::vector<double> temp_stack_event_img_nonuni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> temp_aps_irr_deblur_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	
	long int nonuni_counter = 0;

	long long total_events_decode = 0;
	long long total_events_encode = 0;
	long long total_runtime_encode = 0;
	long long total_runtime_decode = 0;

	double temp_stacked_num_counter_nonuni = 0;

	std::vector<double> crf_exposure;
	std::vector<double> crf_irr;
	std::vector<double> aps_irr_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<int> aps_img = std::vector<int>(DAVIS346_IMAGE_PIXNUM, 0);

	// int dn_upthresh_;
	// int dn_downthresh_;
	double dvs_c_pos_;
	double dvs_c_neg_;
	std::string aps_irr_scenemap_saved_path;
	bool loadOnceFlag_ = true;

	//previous generated HDR scenemap
	std::vector<double> aps_irr_scenemap = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);	

	// variable for event runtime processing
	std::vector<double> ev_irr_runtime_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> stacked_ev_irr_runtime_exp = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);
	std::vector<double> stacked_runtime_counter_uni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);
	double stacked_runtime_counter_nonuni = 0;


	std::vector<double> dynamic_runtime_counter = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);

	std::array<std::array<double, 2>, MAX_SIZE * DAVIS346_IMAGE_PIXNUM> stacked_runtime_counter_nonuni_list;

	std::array<std::array<std::array<double,3>,MAX_SIZE>,DAVIS346_IMAGE_PIXNUM> ev_irr_exp_runtime_list;



	// variable for irrdiance estimation
	std::vector<double>  est_ev_irr_exp_uni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double>  est_ev_irr_exp_nonuni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> est_ev_pixcount_uni = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);

	std::vector<long> runtime_list;


	double est_ev_pixcount_nonuni = 0;

	// variable for irrdiance estimation, aps side
	std::vector<double> aps_irr_log_list_valid;
	std::vector<double> aps_irr_log_list_invalid;
	std::vector<double> aps_irr_latest_est_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
	std::vector<double> aps_irr_deblur_log = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);

	// variable for contineous irridance estimation
	std::vector<double> latest_irr_exp = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);
	std::vector<double> prev_latest_irr_exp;
	std::vector<double> current_latest_irr_exp = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 1);

public:
	SceneIrrdianceEst() {
		outputs.infoNode("deblur").create<dv::Config::AttributeType::STRING>("source", "deblur image for display", {0, 8192},
			dv::Config::AttributeFlags::READ_ONLY | dv::Config::AttributeFlags::NO_EXPORT,
			"Description of the first origin of the data");
	}
	static void initTypes(std::vector<dv::Types::Type> &types) {
		types.push_back(dv::Types::makeTypeDefinition<VectorPack, DoubleVector>("A double vector."));
	}

	static void initInputs(dv::InputDefinitionList &in) {
		in.addFrameInput("frames");
		in.addTriggerInput("triggers");
		in.addEventInput("events");
	}
	static void initOutputs(dv::OutputDefinitionList &out) {
		out.addOutput("deblur", VectorPack::TableType::identifier);

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
			// std::cout<<"------------------------------------------------"<<std::endl;
			// std::cout<<"loading finished, crf size is:"<<crf.size()<<std::endl;
			// std::cout<<"------------------------------------------------"<<std::endl;
			myfile.close();
		}
		return vec;
	}

	void writing_mat_to_file(std::string file_path, const std::vector<double> &vector_to_save)
	{
		std::ofstream myfile;
		myfile.open(file_path);
		
		for(int j = 0; j < vector_to_save.size(); j++)
		{
			myfile<<vector_to_save[j]<<std::endl;
		}
		myfile.close();
	}

	static const char *initDescription() {
		return ("This is the first part of the pipeline, generate the scene irrdiance map.");
	}

	static void initConfigOptions(dv::RuntimeConfig &config) {
		config.add("dvs_c_pos",
			dv::ConfigOption::doubleOption(
				"DVS contrast positive.", 0.2609, 0, 10));
		config.add("dvs_c_neg",
			dv::ConfigOption::doubleOption(
				"DVS contrast negative.", -0.2415, -10, 0));
		config.add("push_or_trigger",
			dv::ConfigOption::boolOption(
				"Select trigger mode or push mode, trigger is inaccurate, true for trigger", true));
	}
	void configUpdate() override {
		// get parameter
		// dn_upthresh_ = config.getInt("dn_upthresh");
		// dn_downthresh_ = config.getInt("dn_downthresh");
		dvs_c_pos_ = config.getDouble("dvs_c_pos");
		dvs_c_neg_ = config.getDouble("dvs_c_neg");
		push_or_trigger = config.getBool("push_or_trigger");
		if(loadOnceFlag_)
		{
			loadOnceFlag_ = false;
			crf_exposure = loading_vec_from_file("/home/eleboss/Documents/ev_runtime/crf/crf.txt");
			aps_irr_scenemap_saved_path = "/home/eleboss/Documents/ev_runtime/dv/vis_data/aps_irr_scenemap.txt";
			aps_irr_scenemap = loading_vec_from_file(aps_irr_scenemap_saved_path);
			for (int i=0; i<aps_irr_scenemap.size(); i++) 
				prev_latest_irr_exp.push_back(exp(aps_irr_scenemap[i])); 
		}

	}

	void run() override 
	{
		auto trigger = inputs.getTriggerInput("triggers").data();
		if(trigger)
		{
			if (trigger.front().type == dv::TriggerType::APS_FRAME_START)
			{
				frontEndSwitch = true;
				trig_aps_start_ts = trigger.front().timestamp/TIME_SCALE;
			}
			else if(trigger.front().type == dv::TriggerType::APS_FRAME_END)
			{
				frontEndSwitch = false;
				trig_aps_end_ts = trigger.front().timestamp/TIME_SCALE;
			}
		}

		double exposure_time = 0;
		// get data
		auto frame = inputs.getFrameInput("frames").data();
		if(frame && !frameReceiveFlag_)
		{
			frame_ts = double(frame.timestamp())/TIME_SCALE;
			frame_ts_frStart = double(frame.timestampStartOfFrame())/TIME_SCALE;
			frame_ts_frEnd = double(frame.timestampEndOfFrame())/TIME_SCALE;
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
				int pix_dn = frame.pixels()[j];
				aps_irr_log[j] = crf_irr[pix_dn];
				aps_img[j] = pix_dn;
			}
			frameReceiveFlag_ = true;
		}



		auto events = inputs.getEventInput("events").events();
		if(events)
		{
			double latest_event_time = double(events[-1].timestamp())/TIME_SCALE;
			double first_event_time = double(events[0].timestamp())/TIME_SCALE;
			///////////////////////////
			// utilize trigger
			// int ev_counter = 0;
			auto start = std::chrono::high_resolution_clock::now();
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
				// push_back comsume a lot amount of time, I am trying to use trigger to boost speed but current trigger is not synced, thus inaccurate.
				// future work will try to use module synced by LUCA.
				int runtime_counter = dynamic_runtime_counter[pix_index];
				ev_irr_exp_runtime_list[pix_index][runtime_counter] = {ev_t_rt, ev_irr_exp_rt, stacked_runtime_counter_nonuni};
				stacked_runtime_counter_nonuni_list[nonuni_counter] = {ev_t_rt, stacked_runtime_counter_nonuni};
				dynamic_runtime_counter[pix_index] += 1;
				nonuni_counter += 1;

			}


			if(latest_event_time > frame_ts_frEnd && frameReceiveFlag_)
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
				
				auto elapsed = std::chrono::high_resolution_clock::now() - start;
				long long runtime_duration = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
				total_events_decode += nonuni_counter;
				total_runtime_decode += runtime_duration;
				double event_rate_decode = total_events_decode / (double(total_runtime_decode)/TIME_SCALE);
				// log.info << "total_runtime_decode: " << total_runtime_decode<< "event rate decode: "<< event_rate_decode << dv::logEnd;

				//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// clear parts of the container 
				stacked_runtime_counter_nonuni = 0;
				std::fill(ev_irr_runtime_log.begin(), ev_irr_runtime_log.end(), 0);
				std::fill(stacked_ev_irr_runtime_exp.begin(), stacked_ev_irr_runtime_exp.end(), 1);
				nonuni_counter = 0;
				dynamic_runtime_counter = std::vector<double>(DAVIS346_IMAGE_PIXNUM, 0);
				//////////////////////////////////////////
				auto out_deblur  = outputs.getVectorOutput<VectorPack, DoubleVector>("deblur").data();
				DoubleVector outFrame_deblur;
				outFrame_deblur.timestamp = frame_ts * TIME_SCALE;
				outFrame_deblur.timestampStartOfExposure = frame_ts_expStart * TIME_SCALE;
				outFrame_deblur.timestampEndOfExposure = frame_ts_expEnd * TIME_SCALE;
				outFrame_deblur.timestampStartOfFrame = frame_ts_frStart * TIME_SCALE;
				outFrame_deblur.timestampEndOfFrame = frame_ts_frEnd * TIME_SCALE;
				outFrame_deblur.pixels = aps_irr_deblur_log;
				out_deblur.push_back(outFrame_deblur);
				out_deblur.commit();
				out_deblur.clear();
			}
			


		}
	}
};

registerModuleClass(SceneIrrdianceEst)
