<?php
namespace library\tree;
use library\dom\object;
use library\dom\elements\components\text;
use library\dom\elements\void;
use library\tree\leaf;
use library\tree\branch;

class nodeFactory{
    public static function create(object $object){
        if($object instanceof text || $object instanceof void){
            return new leaf($object);
        }
        return new branch($object);
    }
}