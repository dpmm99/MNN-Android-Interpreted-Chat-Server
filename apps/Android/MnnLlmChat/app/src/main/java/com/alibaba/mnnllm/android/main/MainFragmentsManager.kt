package com.alibaba.mnnllm.android.main

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.alibaba.mnnllm.android.benchmark.BenchmarkFragment
import com.alibaba.mnnllm.android.hotspot.ChatServerFragment
import com.alibaba.mnnllm.android.modelist.ModelListFragment
import com.alibaba.mnnllm.android.modelmarket.ModelMarketFragment
import com.alibaba.mnnllm.android.utils.Searchable
import com.alibaba.mnnllm.android.widgets.BottomTabBar

/**
 * Manages the main fragments in MainActivity, including creation, transactions,
 * and state restoration.
 *
 * @param activity The host activity.
 * @param containerId The ID of the container where fragments will be placed.
 * @param bottomNav The BottomTabBar view to control fragment switching.
 * @param listener A listener to communicate UI update events back to the activity.
 * @param modelListChangeListener Optional listener for model list changes (e.g. model deleted); activity uses it to notify ModelMarketFragment.
 */
class MainFragmentManager(
    private val activity: AppCompatActivity,
    private val containerId: Int,
    private val bottomNav: BottomTabBar,
    private val listener: FragmentLifecycleListener,
    private val modelListChangeListener: com.alibaba.mnnllm.android.modelist.OnModelListChangeListener? = null
) {
    private var modelListFragment: ModelListFragment? = null
    private var modelMarketFragment: ModelMarketFragment? = null
    private var benchmarkFragment: BenchmarkFragment? = null
    private var chatServerFragment: ChatServerFragment? = null
    var activeFragment: Fragment? = null

    /** * An interface for the manager to communicate important events back to the hosting Activity. * This lets Activity can respond to Fragment changes to updateits own UI (e.g., Toolbar title).*/
    interface FragmentLifecycleListener {
        fun onTabChanged(newTab: BottomTabBar.Tab)
    }

    /**
     * Initializes the fragments. Call this in Activity's onCreate.
     * This method handles both initial creation and restoration from a saved state.
     */
    fun initialize(savedInstanceState: Bundle?) {
        if (savedInstanceState == null) {
            modelListFragment = ModelListFragment()
            modelMarketFragment = ModelMarketFragment()
            benchmarkFragment = BenchmarkFragment()
            chatServerFragment = ChatServerFragment()

            // On a warm start the launcher may recreate the Activity with no savedInstanceState,
            // but processLastTab (a JVM static field) still holds the last-selected tab from
            // this process's lifetime. On a true cold start processLastTab is null → LOCAL_MODELS.
            val targetFragment: Fragment = when (processLastTab) {
                BottomTabBar.Tab.MODEL_MARKET -> modelMarketFragment!!
                BottomTabBar.Tab.BENCHMARK -> benchmarkFragment!!
                BottomTabBar.Tab.CHAT_SERVER -> chatServerFragment!!
                else -> modelListFragment!!
            }

            activity.supportFragmentManager.beginTransaction()
                .add(containerId, chatServerFragment!!, TAG_CHAT_SERVER)
                .add(containerId, benchmarkFragment!!, TAG_BENCHMARK)
                .add(containerId, modelMarketFragment!!, TAG_MARKET)
                .add(containerId, modelListFragment!!, TAG_LIST)
                .also { tx ->
                    if (targetFragment !== chatServerFragment) tx.hide(chatServerFragment!!)
                    if (targetFragment !== benchmarkFragment) tx.hide(benchmarkFragment!!)
                    if (targetFragment !== modelMarketFragment) tx.hide(modelMarketFragment!!)
                    if (targetFragment !== modelListFragment) tx.hide(modelListFragment!!)
                }
                .commit()

            activeFragment = targetFragment
        } else {
            modelListFragment = activity.supportFragmentManager.findFragmentByTag(TAG_LIST) as? ModelListFragment
            modelMarketFragment = activity.supportFragmentManager.findFragmentByTag(TAG_MARKET) as? ModelMarketFragment
            benchmarkFragment = activity.supportFragmentManager.findFragmentByTag(TAG_BENCHMARK) as? BenchmarkFragment
            chatServerFragment = activity.supportFragmentManager.findFragmentByTag(TAG_CHAT_SERVER) as? ChatServerFragment

            val activeTag = savedInstanceState.getString(SAVED_STATE_ACTIVE_TAG)
            if (activeTag != null) {
                activeFragment = activity.supportFragmentManager.findFragmentByTag(activeTag)
            } else {
                activeFragment = listOfNotNull(modelListFragment, modelMarketFragment, benchmarkFragment, chatServerFragment)
                    .find { !it.isHidden }
            }
        }

        setupTabListener()
        modelListFragment?.onModelListChangeListener = modelListChangeListener
        val initialTab = getTabForFragment(activeFragment)
        // Keep processLastTab in sync so that any subsequent recreation within this
        // process (e.g. another launcher tap) also lands on the correct tab.
        processLastTab = initialTab
        bottomNav.select(initialTab)
        listener.onTabChanged(initialTab)
    }

    /**
     * Saves the current active fragment's tag to the bundle. Call this in Activity's onSaveInstanceState.
     */
    fun onSaveInstanceState(outState: Bundle) {
        if (activeFragment?.tag != null) {
            outState.putString(SAVED_STATE_ACTIVE_TAG, activeFragment!!.tag)
        }
    }

    /**
     * Returns the currently active fragment, which might implement the Searchable interface.
     */
    fun getActiveSearchableFragment(): Searchable? {
        return activeFragment as? Searchable
    }

    private fun setupTabListener() {
        bottomNav.setOnTabSelectedListener { tab ->
            Log.d(TAG, "Tab selected: $tab")

            val targetFragment = when (tab) {
                BottomTabBar.Tab.LOCAL_MODELS -> modelListFragment
                BottomTabBar.Tab.MODEL_MARKET -> modelMarketFragment
                BottomTabBar.Tab.BENCHMARK -> benchmarkFragment
                BottomTabBar.Tab.CHAT_SERVER -> chatServerFragment
            }

            if (targetFragment != null && activeFragment != targetFragment) {
                switchFragment(targetFragment)
                listener.onTabChanged(tab) //Notify Activity
            }
        }
    }

    private fun switchFragment(targetFragment: Fragment) {
        processLastTab = getTabForFragment(targetFragment)
        activity.supportFragmentManager.beginTransaction().apply {
            if (activeFragment != null) {
                hide(activeFragment!!)
            }
            show(targetFragment)
            commitNow()
        }
        activeFragment = targetFragment
    }

    private fun getTabForFragment(fragment: Fragment?): BottomTabBar.Tab {
        return when (fragment) {
            is ModelMarketFragment -> BottomTabBar.Tab.MODEL_MARKET
            is BenchmarkFragment -> BottomTabBar.Tab.BENCHMARK
            is ChatServerFragment -> BottomTabBar.Tab.CHAT_SERVER
            else -> BottomTabBar.Tab.LOCAL_MODELS
        }
    }

    /**
     * Notify ModelMarketFragment that the set of downloaded models has changed (e.g. after delete in ModelListFragment).
     */
    fun notifyModelMarketDownloadedModelsChanged() {
        modelMarketFragment?.onDownloadedModelsChanged()
    }

    companion object {
        private const val TAG = "MainFragmentManager"
        private const val TAG_LIST = "list"
        private const val TAG_MARKET = "market"
        private const val TAG_BENCHMARK = "benchmark"
        private const val TAG_CHAT_SERVER = "chat_server"
        private const val SAVED_STATE_ACTIVE_TAG = "active_fragment_tag"

        /**
         * Tracks the last active tab at the process level.
         *
         * This is a JVM static field initialized to null. It is null at cold start
         * (fresh process) so the default LOCAL_MODELS tab is used. On a warm start,
         * the process survives and this retains the previously selected tab, so a newly
         * created Activity instance restores the correct tab even when savedInstanceState
         * is null (as happens when the launcher recreates the Activity via
         * FLAG_ACTIVITY_RESET_TASK_IF_NEEDED).
         */
        private var processLastTab: BottomTabBar.Tab? = null
    }
}