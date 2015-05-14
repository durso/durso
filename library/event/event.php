<?php

/**
 * Description of event
 *
 * @author durso
 */

namespace library\event;
use library\layout\elements\element;

class event {
    private static $events = array();
    private static $eventList = array("click");
    
    public static function trigger($id,$event, $args = array())
    {
        if(isset(self::$events[$id][$event]))
        {
            foreach(self::$events[$id][$event] as $callback)
            {
                $callback->$event($callback);
            }
        }

    }
    public static function register(element $element, $event){
        try{
            if(self::isEvent($event)){
                self::$events[$element->getId()][$event][] = $element;
            } else{
                throw new \Exception("Event not supported");
            }
        } catch (\Exception $e) {
            die("not an event: ".$e->getMessage());    
        }
    }
    private static function isEvent($event){
        return in_array($event, self::$eventList);
    }
}
